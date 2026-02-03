#!/bin/bash
# Minimal SFT training script for Task1 (x3, multi-image) using train_mem.py.

set -e

REPO_ROOT="/local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer"
cd "$REPO_ROOT"

echo "=========================================="
echo "SFT training on Task1 (x3, multi-image)"
echo "=========================================="

PYTHON="/local_data/ky2738/envs/snpp2msg-rast/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "❌ Python not found: $PYTHON"
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUT_DIR="./tmp/spatial_tasks_x3"
MODEL="./runs/train/RoboRefer-2B-Depth-Align"
TRAIN_JSON="$OUT_DIR/t1_train_multi.json"

if [ ! -f "$TRAIN_JSON" ]; then
  echo "⚠️  $TRAIN_JSON not found, building spatial_tasks_x3 first ..."
  "$PYTHON" tools/build_spatial_tasks_x3.py \
    --data_root /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3 \
    --out_dir "$OUT_DIR" \
    --mode anchor \
    --anchor_k 1 \
    --emit_concat 1
fi

OUTPUT_DIR="./runs/train/mv_consistency_sft_x3"
mkdir -p "$OUTPUT_DIR"

# FREEZE_VISION=1 will set tune_vision_tower=False
if [ "${FREEZE_VISION:-0}" = "1" ]; then
  TUNE_VISION="False"
  echo "Using FREEZE_VISION=1 -> tune_vision_tower=False"
else
  TUNE_VISION="True"
  echo "Using FREEZE_VISION=0 -> tune_vision_tower=True"
fi

echo ""
echo "Launching train_mem.py ..."

set -x
"$PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=29533 \
  llava/train/train_mem.py \
  --deepspeed scripts/zero3.json \
  --model_name_or_path "$MODEL" \
  --chat_template qwen2 \
  --data_mixture "" \
  --data_path "$TRAIN_JSON" \
  --image_folder / \
  --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
  --depth_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
  --mm_vision_select_feature cls_patch \
  --mm_projector mlp_downsample_3x3_fix \
  --depth_projector mlp_downsample_3x3_fix \
  --enable_depth False \
  --use_depth_tower False \
  --tune_vision_tower "$TUNE_VISION" \
  --tune_mm_projector True \
  --tune_language_model True \
  --tune_depth_tower False \
  --tune_depth_projector False \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio resize \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --learning_rate 1e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --report_to none \
  2>&1 | tee "$OUTPUT_DIR/train.log"
set +x

echo ""
echo "=========================================="
echo "✅ SFT training (Task1 x3) finished."
echo "=========================================="

