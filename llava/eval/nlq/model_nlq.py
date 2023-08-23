import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from IPython import embed
from PIL import Image
import math
import decord

class NLQ_DATASET(torch.utils.data.Dataset):
    def __init__(self, args):
        videos = json.load(open(args.metadata_path))['videos']
        self.video_folder = args.video_folder
        self.metadata = []
        self.load_all_frames = args.mean_frames
        for video in videos:
            for clip in video['clips']:
                clip_path = os.path.join(self.video_folder, f"{clip['clip_uid']}.mp4")
                if not os.path.exists(clip_path):
                    continue
                for annotation in clip['annotations']:
                    for query in annotation['language_queries']:
                        self.metadata.append({
                            'clip_path': clip_path,
                            'start_frame': query['video_start_frame'] - clip['video_start_frame'],
                            'end_frame': query['video_end_frame'] - clip['video_start_frame'],
                            'query': query['query'],
                            'clip_start_sec': query['clip_start_sec'],
                            'clip_end_sec': query['clip_end_sec'],
                        })
                   

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        clip_metadata = self.metadata[idx]
        clip_path = clip_metadata['clip_path']
        start_frame = clip_metadata['start_frame']
        end_frame = clip_metadata['end_frame']
        query = clip_metadata['query']
        if self.load_all_frames:
            frames = self.get_frames(clip_path, start_frame, end_frame)
        else:
            frames = self.get_three_frames(clip_path, start_frame, end_frame)
        return {
            'frames': frames,
            'query': query,
            'clip_uid': os.path.basename(clip_path).split('.')[0],
            'clip_start_sec': clip_metadata['clip_start_sec'],
            'clip_end_sec': clip_metadata['clip_end_sec'],
        }
    
    @staticmethod
    def get_frames(video_path, start_frame: int, end_frame: int):
        vr = decord.VideoReader(video_path)
        max_frames = 150
        frame_ids = list(range(start_frame, end_frame))
        frames = []
        if len(frame_ids) > max_frames:
            stride = math.ceil(len(frame_ids) / max_frames)
            frame_ids = frame_ids[::stride]
        for frame_id in frame_ids:
            frame = vr.get_batch([frame_id]).asnumpy()[0]
            image = Image.fromarray(frame)
            frames.append(image)
        return frames

    @staticmethod
    def get_three_frames(video_path, start_frame: int, end_frame: int):
        vr = decord.VideoReader(video_path)
        frames = []
        for frame_id in [start_frame, (start_frame + end_frame) // 2, end_frame]:
            frame = vr.get_batch([frame_id]).asnumpy()[0]
            image = Image.fromarray(frame)
            frames.append(image)
        return frames

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.get_model().vision_tower.to(args.vision_device)
    dataset = NLQ_DATASET(args)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    dataset[0]
    for data in tqdm(dataset):
        idx = data['clip_uid']
        qs = data['query']
        cur_prompt = qs
        frames = data['frames']
        if model.config.mm_use_im_start_end:
            if not args.mean_frames:
                for _ in range(len(frames)):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            if not args.mean_frames:
                for _ in range(len(frames)):
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        frame_tensor_list = [image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda() for frame in frames]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                # images=frame_tensor_list.unsqueeze(0).half().cuda(),
                images = frame_tensor_list,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"clip_uid": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "clip_start_sec": data['clip_start_sec'],
                                    "clip_end_sec": data['clip_end_sec'],
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--video-folder", type=str, default="/data/dongwang/ego4d/v2/clips")
    parser.add_argument("--metadata-path", type=str, default="/home/dongwang/LLaVA/playground/data/nlq/nlq_val.json")
    parser.add_argument("--mean-frames", action='store_true', default=False)
    args = parser.parse_args()
    args.device = torch.device('cuda', index=0)
    args.vision_device = torch.device('cuda', index=1)

    eval_model(args)
