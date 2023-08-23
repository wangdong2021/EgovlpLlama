# export CUDA_VISIBLE_DEVICES=7
python llava/train/train_egovlp.py \
    --model_name_or_path /data/dongwang/vicuna-7b-v1.3 \
    --tune_mm_mlp_adapter True \
    --mm_video_select_layer -2 \
    --mm_use_im_start_end True \
    --meta_dir /home/dongwang/EgoVLP/subset \
    --data_dir /data/dongwang/instance_videos \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to wandb
# CUDA_VISIBLE_DEVICES=0,7 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
