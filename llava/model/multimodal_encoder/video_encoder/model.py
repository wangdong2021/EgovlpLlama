import os
import pdb

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from IPython import embed
from .video_transformer import SpaceTimeTransformer
import torch.distributed as dist
import numpy as np

def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp: # this
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict

class VideoEncoder(nn.Module):
    def __init__(self,
                 video_params: dict,
                 load_temporal_fix='zeros',
                ):
        super().__init__()
        self.video_params = video_params
        self.load_temporal_fix = load_temporal_fix
        load_checkpoint = video_params.mm_load_checkpoint
        pretrained = video_params.mm_pretrained_vit
        num_frames = video_params.mm_num_frames
        time_init = video_params.mm_time_init

        vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained).state_dict()
        model = SpaceTimeTransformer(num_frames=num_frames, time_init=time_init)

        model.head = nn.Identity()
        model.pre_logits = nn.Identity()

        # load vision transformer weights for initialization
        if load_checkpoint in ["", None]:
            vit_checkpoint = vit_model

            for key in list(vit_checkpoint.keys()):
                if ('mlp' in key and 'mlp_ego' not in key and 'mlp_third' not in key) and 'blocks' in key:
                    ego_key = key.replace('mlp', 'mlp_ego')
                    third_key = key.replace('mlp', 'mlp_third')
                    vit_checkpoint[ego_key] = vit_checkpoint[key]
                    vit_checkpoint[third_key] = vit_checkpoint[key]
                if ('norm2' in key and 'norm2_ego' not in key and 'norm2_third' not in key) and 'blocks' in key:
                    ego_key = key.replace('norm2', 'norm2_ego')
                    third_key = key.replace('norm2', 'norm2_third')
                    vit_checkpoint[ego_key] = vit_checkpoint[key]
                    vit_checkpoint[third_key] = vit_checkpoint[key]
            print('begin to state_dict_data_parallel_fix')
            new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
            print('done state_dict_data_parallel_fix')
            info = model.load_state_dict(new_vit_dict, strict=False)
            print(info)
            print('done info')
        self.video_model = model

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        if load_checkpoint not in ["", None]:
            local_rank = int(os.environ['LOCAL_RANK'])  # fixed by qinghong.
            print(f'load {load_checkpoint}')
            checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(local_rank))
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            if 'video_model.cls_token' in new_state_dict \
                    and 'video_model.ego_cls_token' not in new_state_dict \
                    and 'video_model.third_cls_token' not in new_state_dict:
                print('video_model.cls_token -> video_model.ego_cls_token and video_model.third_cls_token')
                new_state_dict['video_model.ego_cls_token'] = new_state_dict['video_model.cls_token']
                new_state_dict['video_model.third_cls_token'] = new_state_dict['video_model.cls_token']
            
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            new_state_dict.pop('video_model.cls_token', '')
            info = self.load_state_dict(new_state_dict, strict=False)
            print(info)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def set_device(self, device):
        self.device = device

    def forward(self, videos):
        first_video_embeds = self.compute_video(videos, is_ego=True) 
        third_video_embeds = self.compute_video(videos, is_ego=False)
        video_embeds = torch.stack([first_video_embeds, third_video_embeds], dim=1)
        video_embeds = video_embeds.flatten(0, 1)
        return video_embeds

    def compute_video(self, video_data, is_ego=True):
        video_embeddings = self.video_model.forward_features(video_data, is_ego=is_ego, select_layer=self.video_params.mm_video_select_layer)
        if self.video_params.mm_video_select_feature == "patch":
            video_embeddings = video_embeddings[:, 1:]
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params.mm_num_frames
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

    @property
    def hidden_size(self):
        return self.video_model.embed_dim