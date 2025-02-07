export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
#export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0,1,2,3,4,5,6,7"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("5e-5")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("5000")
HEIGHT_BUCKETS="960"
WIDTH_BUCKETS="1360"
FRAME_BUCKETS="49"
MAX_NUM_FRAMES="49"
LORA_RANK=256

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_8.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="/data/cogvid_preproc_crop_latents_v15"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="/data/cogvideox-factory-animl/runs/cogvideox-lora-v1_guide_crop_49__steps_${steps}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_frames_to_video_lora.py \
          --pretrained_model_name_or_path /data/cogvideox-factory-animl/ckpt/CogVideoX-5b-I2V_guide \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --id_token BW_STYLE \
          --height_buckets $HEIGHT_BUCKETS \
          --width_buckets $WIDTH_BUCKETS \
          --frame_buckets $FRAME_BUCKETS \
          --max_num_frames $MAX_NUM_FRAMES \
          --dataloader_num_workers 16 \
          --pin_memory \
          --validation_prompt \"Side view of a nice sneaker, while camera trajectory is toward the right.:::Front view of a nice sneaker, while camera trajectory is toward the left.\"
          --validation_images \"/data/cogvideox-factory-animl/assets/tests/videos/sneaker_side2right.mp4:::/data/cogvideox-factory-animl/assets/tests/videos/sneaker_front2left.mp4\"
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --rank $LORA_RANK \
          --lora_alpha $LORA_RANK \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --train_batch_size 4 \
          --max_train_steps $steps \
          --checkpointing_steps 250 \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 400 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --noised_image_dropout 0.05 \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --load_tensors \
          --condition_frames_dropout 0.75 \
          --ignore_learned_positional_embeddings \
          --resume_from_checkpoint latest \
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
