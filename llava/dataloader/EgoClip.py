import os
import json
from typing import Any
import pandas as pd
import json
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import random
import decord
from torchvision.transforms._transforms_video import RandomResizedCropVideo, NormalizeVideo, RandomHorizontalFlipVideo
from llava.constants import DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, IGNORE_INDEX
from dataclasses import dataclass
import transformers
from typing import Dict, Sequence

class EgoClip(Dataset):

    full_set_file_name = 'egoclip.csv'
    subset_file_name = 'egoclip_subset_200.csv'
    third_caption_file_name = 'third_caption_train_200000.json'
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    user_prompt_with_third_caption = [
        "USER: {first_video_feature} is a first-person perspective video of an event, while {third_video_feature} is a third-person perspective video of the same event. Please provide a brief description for each of these perspectives. ASSISTANT: ",
        "USER: Give a brief description for the first-person perspective video, {first_video_feature}, and the third-person perspective video, {third_video_feature}, of the same event. ASSISTANT: ",
        "USER: Describe the unique viewpoint offered by {first_video_feature}, a first-person perspective video of the event, and compare it to the perspective provided by {third_video_feature}, a third-person perspective video. ASSISTANT: ",
        "USER: What distinguishes {first_video_feature}, a first-person perspective video, from {third_video_feature}, a third-person perspective video, of the same event? Provide a concise description for each perspective. ASSISTANT: ",
        "USER: Discuss the immersive nature of {first_video_feature}, a first-person perspective video, and contrast it with {third_video_feature}, a video capturing the event from a third-person perspective. ASSISTANT: ",
        "USER: Provide a brief description for both {first_video_feature}, a first-person perspective video, and {third_video_feature}, a third-person perspective video, highlighting the unique qualities of each perspective. ASSISTANT: ",
        "USER: Describe the distinct vantage points portrayed in {first_video_feature}, a first-person perspective video, versus {third_video_feature}, a third-person perspective video, capturing the same event. ASSISTANT: "
        "USER: Explain how {first_video_feature}, a first-person perspective video, and {third_video_feature}, a third-person perspective video, offer different insights into the same event. Provide a summary for each perspective. ASSISTANT: ",
        "USER: Contrast {first_video_feature}, a first-person perspective video, with {third_video_feature}, a third-person perspective video, of the same event, highlighting the subjective and objective aspects of each perspective. ASSISTANT: ",
        "USER: Discuss the experiential and observational aspects of {first_video_feature}, a first-person perspective video, and {third_video_feature}, a third-person perspective video, capturing the same event. ASSISTANT: ",
        "USER: Provide a brief summary for {first_video_feature}, a first-person perspective video, and {third_video_feature}, a third-person perspective video, showcasing different angles of the same event. ASSISTANT: "
    ]
    answer_prompt_with_third_caption = "for the first-person perspective video: {first_caption}; for the third-person perspective video: {third_caption} </s>"
    
    user_prompt_without_third_caption = [
        "USER: {first_video_feature} is a first-person perspective video of an event. Please provide a brief description for this video. ASSISTANT: ",
        "USER: Describe the event portrayed in {first_video_feature}, a first-person perspective video, providing a concise summary. ASSISTANT: ",
        "USER: Give a brief overview of the footage captured in {first_video_feature}, a first-person perspective video, highlighting the unique perspective it offers. ASSISTANT: ",
        "USER: Provide a short description for {first_video_feature}, a first-person perspective video, showcasing the event from the participant's point of view. ASSISTANT: ",
        "USER: Explain the viewpoint presented in {first_video_feature}, a first-person perspective video, and how it enhances the viewer's understanding of the event. ASSISTANT: ",
        "USER: Summarize the content of {first_video_feature}, a first-person perspective video, capturing the event from a personal and immersive angle. ASSISTANT: ",
        "USER: Highlight the immersive elements captured in {first_video_feature}, a first-person perspective video, that allows viewers to experience the event firsthand. ASSISTANT: ",
        "USER: Discuss the intimate nature of {first_video_feature}, a first-person perspective video, offering viewers a unique and personal insight into the event. ASSISTANT: ",
        "USER: Describe the emotions and sensations depicted in {first_video_feature}, a first-person perspective video, enhancing the viewer's connection to the event. ASSISTANT: ",
        "USER: Provide a concise overview of {first_video_feature}, a first-person perspective video, allowing viewers to witness the event through the eyes of a participant. ASSISTANT: ",
        "USER: Explain how {first_video_feature}, a first-person perspective video, enhances the viewer's engagement with the event, providing an immersive visual experience. ASSISTANT: "
    ]
    answer_prompt_without_third_caption = "{first_caption} </s>"

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        """the initialization function for EgoClip dataset

        Args:
            data_args (_type_): 
                data_dir: str, the directory of the dataset
                meta_dir: str, the directory of the metadata
                use_third_caption: bool, whether to use the third caption
                num_frames: int, the number of frames to sample from each video
                mm_use_im_start_end: bool, whether to use the start and end tokens for image
                full_set: bool, whether to use the full set or the subset

            tokenizer (transformers.PreTrainedTokenizer): 
                the tokenizer of LLM
        """
        self.data_dir = data_args.data_dir
        self.meta_dir = data_args.meta_dir
        self.use_third_caption = data_args.use_third_caption
        self.num_frames = data_args.num_frames
        self.use_start_end = data_args.mm_use_im_start_end
        self.tokenizer = tokenizer
        self.transforms = EgoClip.init_video_transform_dict()
        if data_args.full_set:
            meta_file = EgoClip.full_set_file_name
        else:
            meta_file = EgoClip.subset_file_name
        
        self.metadata = pd.read_csv(os.path.join(self.meta_dir, meta_file), sep='\t')
        self.frame_sample = 'rand'
        if self.use_third_caption:
            self.caption_json_file = os.path.join(self.meta_dir, EgoClip.third_caption_file_name)
            with open(self.caption_json_file, 'r', encoding='UTF-8') as f:
                self.caption_json_dict = json.load(f)

    @staticmethod
    def init_video_transform_dict(
            input_res=224,
            randcrop_scale=(0.5, 1.0),
            color_jitter=(0, 0, 0),
            norm_mean=(0.485, 0.456, 0.406),
            norm_std=(0.229, 0.224, 0.225),
            use_clip=False
    ):
        if not use_clip:
            normalize = NormalizeVideo(mean=norm_mean, std=norm_std)
        else:
            CLIP_DEFAULT_MEAN = (0.4815, 0.4578, 0.4082)
            CLIP_DEFAULT_STD = (0.2686, 0.2613, 0.2758)
            normalize = NormalizeVideo(mean=CLIP_DEFAULT_MEAN, std=CLIP_DEFAULT_STD)

        return transforms.Compose([
            RandomResizedCropVideo(input_res, scale=randcrop_scale),
            RandomHorizontalFlipVideo(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ])

    def sample_frames_start_end(self, video_length):
        acc_samples = min(self.num_frames, video_length)
        intervals = np.linspace(start=0, stop=video_length, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interval in enumerate(intervals[:-1]):
            ranges.append((interval, intervals[idx + 1]))
        if self.frame_sample == 'rand':
            frame_idxs = []
            for x in ranges:
                if x[0] == x[1]:
                    frame_idxs.append(x[0])
                else:
                    frame_idxs.append(random.choice(range(x[0], x[1])))

            if len(ranges) == 0:
                frame_idxs.append(intervals[0])
        elif self.frame_sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError
        return frame_idxs

    def _get_video_frames(self, video_path):
        video_data = decord.VideoReader(video_path, ctx=decord.cpu())
        video_length = len(video_data)
        frame_idxs = self.sample_frames_start_end(video_length)
        frames = video_data.get_batch(frame_idxs).asnumpy()
        frames = torch.from_numpy(frames).permute(0,3,1,2).float() / 255
        frames = frames.transpose(0, 1)
        frames = self.transforms(frames)
        frames = frames.transpose(0, 1)
        if frames.shape[0] == self.num_frames:
            return frames
        else:
            ret_frames = torch.zeros([self.num_frames, frames.shape[1], frames.shape[2], frames.shape[3]])
            ret_frames[:frames.shape[0]] = frames
            ret_frames[frames.shape[0]:] = frames[-1]
            return ret_frames

    def get_third_caption(self, sample):
        narration_source = str(sample['narration_source'])
        narration_ind = str(sample['narration_ind'])
        video_uid = str(sample['video_uid'])
        key = "_".join([narration_source, narration_ind, video_uid])
        third_caption_text = self.caption_json_dict[key]
        return third_caption_text

    def get_input_ids(self, first_caption, third_caption):
        if self.use_start_end:
            video_token = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VID_END_TOKEN
        else:
            video_token = DEFAULT_VIDEO_TOKEN

        if self.use_third_caption and third_caption is not None:
            sentence = EgoClip.system_prompt
            sentence = sentence + random.choice(EgoClip.user_prompt_with_third_caption).format(first_video_feature=video_token, third_video_feature=video_token)
            ignore_part = sentence
            sentence = sentence + EgoClip.answer_prompt_with_third_caption.format(first_caption=first_caption, third_caption=third_caption)
            
        else:
            sentence = EgoClip.system_prompt
            sentence = sentence + random.choice(EgoClip.user_prompt_without_third_caption).format(first_video_feature=video_token)
            ignore_part = sentence
            sentence = sentence + EgoClip.answer_prompt_without_third_caption.format(first_caption=first_caption)
        input_ids = torch.tensor(self.tokenizer(sentence).input_ids)
        labels = input_ids.clone()
        labels[:len(self.tokenizer(ignore_part).input_ids)] = IGNORE_INDEX
        return input_ids, labels

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_path = os.path.join(self.data_dir, sample['video_uid'], f"{sample['narration_source']}_{sample['narration_ind']}.mp4")
        frames = self._get_video_frames(video_path)
        first_caption = sample['clip_text'].replace("#C", "The protagonist in the video is C. ")
        third_caption = self.get_third_caption(sample) if self.use_third_caption else None
        data = {
            "frames": frames,
        }
        data['input_ids'], data['labels'] = self.get_input_ids(first_caption, third_caption)
        return data

@dataclass
class DataCollatorForEgoClip(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        batch['frames'] = torch.stack([instance['frames'] for instance in instances])

        return batch


def make_egoclip_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = EgoClip(data_args=data_args, tokenizer=tokenizer)
    data_collator = DataCollatorForEgoClip(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

if __name__ == "__main__":
    import argparse
    from IPython import embed
    data_args = {
        "data_dir": "/data/dongwang/instance_videos",
        'meta_dir': "/home/dongwang/EgoVLP/subset",
        'use_third_caption': False,
        'num_frames': 4,
        'mm_use_im_start_end': True,
        'full_set': False,
    }
    data_args = argparse.Namespace(**data_args)
    llama_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/data/dongwang/vicuna-7b-v1.3",
        model_max_length=2048, padding_side='right', use_fast=False
    )
    llama_tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
    data_module = make_egoclip_data_module(llama_tokenizer, data_args)
