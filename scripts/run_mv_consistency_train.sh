#!/bin/bash
# 多视角一致性训练脚本

set -e

REPO_ROOT="/local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer"
cd "$REPO_ROOT"

echo "=========================================="
echo "多视角一致性训练"
echo "=========================================="

# 1. 使用上一级目录环境的 Python（二进制直连，不在脚本里管 conda）
echo ""
echo "步骤1: 使用 /local_data/ky2738/envs/snpp2msg-rast/bin/python ..."
PYTHON="/local_data/ky2738/envs/snpp2msg-rast/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "❌ 找不到可执行的 Python: $PYTHON"
    exit 1
fi

echo "Python 可执行: $PYTHON"

# 设置 PYTHONPATH 以便在该环境中找到当前 repo 的 llava 包
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"

# 2. 构建数据集（如已存在则跳过）
echo ""
echo "步骤2: 构建训练数据..."
if [ -f ./tmp/mv_consistency_sft_v2/mv_train.json ] && [ -f ./tmp/mv_consistency_sft_v2/mv_val.json ]; then
    echo "✅ 检测到已有数据集，跳过构建步骤 (tmp/mv_consistency_sft_v2)"
else
    "$PYTHON" tools/build_mv_consistency_sft.py \
      --data_root /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected \
      --out_dir ./tmp/mv_consistency_sft_v2 \
      --mode anchor \
      --anchor_k 1 \
      --neg_ratio 0.0

    if [ $? -ne 0 ]; then
        echo "❌ 数据构建失败！"
        exit 1
    fi

    echo ""
    echo "✅ 数据构建完成"
fi

echo ""

# 3. 启动训练
echo "步骤3: 启动训练..."
"$PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port 29513 \
  llava/train/train_mem.py \
  --deepspeed scripts/zero3.json \
  --model_name_or_path ./runs/train/RoboRefer-2B-Depth-Align \
  --chat_template qwen2 \
  --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
  --depth_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
  --mm_vision_select_feature cls_patch \
  --mm_projector mlp_downsample_3x3_fix \
  --depth_projector mlp_downsample_3x3_fix \
  --enable_depth False \
  --use_depth_tower False \
  --tune_vision_tower True \
  --tune_mm_projector True \
  --tune_language_model True \
  --tune_depth_tower False \
  --tune_depth_projector False \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio dynamic \
  --bf16 True \
  --output_dir ./runs/train/mv_consistency_sft_v2 \
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
  --model_max_length 16384 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --report_to none \
  --data_path ./tmp/mv_consistency_sft_v2/mv_train.json \
  --image_folder /

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
