# CUDA_VISIBLE_DEVICES=1 python ./llava/eval/model_vqa.py \
#     --model-path /home/dongwang/LLaVA/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
#     --answers-file playground/data/nlq/nlq_answers.jsonl \
#     --mean-frames
CUDA_VISIBLE_DEVICES=0,1 python ./llava/eval/nlq/model_nlq.py \
    --model-path /home/dongwang/LLaVA/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3 \
    --answers-file playground/data/nlq/nlq_answers_with_3_frames.jsonl 
    