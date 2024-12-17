#!/bin/bash

MODEL_ID="THUDM/CogVideoX1.5-5B-I2V"

NUM_GPUS=8

# For more details on the expected data format, please refer to the README.
DATA_ROOT="/mnt/data/cogvid_preproc_bottom"  # This needs to be the path to the base directory where your videos are located.
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/mnt/data/cogvid_preproc_bottom_latents"
#HEIGHT_BUCKETS="480 720"
#WIDTH_BUCKETS="720 960"
HEIGHT_BUCKETS="1024"
WIDTH_BUCKETS="1024"
FRAME_BUCKETS="29"
MAX_NUM_FRAMES="29"
MAX_SEQUENCE_LENGTH=226
TARGET_FPS=8
BATCH_SIZE=8
DTYPE=fp16

# To create a folder-style dataset structure without pre-encoding videos and captions
# For Image-to-Video finetuning, make sure to pass `--save_image_latents`
CMD_WITHOUT_PRE_ENCODING="\
  torchrun --nproc_per_node=$NUM_GPUS \
    training/prepare_dataset_dev.py \
      --model_id $MODEL_ID \
      --data_root $DATA_ROOT \
      --caption_column $CAPTION_COLUMN \
      --video_column $VIDEO_COLUMN \
      --output_dir $OUTPUT_DIR \
      --height_buckets $HEIGHT_BUCKETS \
      --width_buckets $WIDTH_BUCKETS \
      --frame_buckets $FRAME_BUCKETS \
      --max_num_frames $MAX_NUM_FRAMES \
      --max_sequence_length $MAX_SEQUENCE_LENGTH \
      --target_fps $TARGET_FPS \
      --batch_size $BATCH_SIZE \
      --use_tiling \
      --dtype $DTYPE
"

CMD_WITH_PRE_ENCODING="$CMD_WITHOUT_PRE_ENCODING --save_image_latents --save_latents_and_embeddings"

# Select which you'd like to run
CMD=$CMD_WITH_PRE_ENCODING

echo "===== Running \`$CMD\` ====="
eval $CMD
echo -ne "===== Finished running script =====\n"
