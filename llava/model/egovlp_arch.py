#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_video_tower

from transformers import LlamaConfig, LlamaModel

class EgovlpConfig(LlamaConfig):
    model_type = "egovlp"

class EgovlpLlamaModel(LlamaModel):
    config_class = EgovlpConfig

    def __init__(self, config: LlamaConfig):
        super(EgovlpLlamaModel, self).__init__(config)

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def initialize_video_modules(self, model_args):
        mm_video_select_layer = model_args.mm_video_select_layer
        mm_video_select_feature = model_args.mm_video_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.video_tower = build_video_tower(model_args)

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = self.video_tower.hidden_size
        self.config.mm_vision_select_layer = mm_video_select_layer
        self.config.mm_vision_select_feature = mm_video_select_feature

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

