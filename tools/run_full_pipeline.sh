#!/bin/bash
# 完整的depth pipeline流程：生成depth -> 构建SFT -> 训练200步

set -e

# ====== 配置 ======
FIVE_FRAMES_ROOT="/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3"
DEPTH_OUTPUT_ROOT="tmp/scannet_depth"  # 使用相对路径避免权限问题
SFT_JSON="tmp/crossview_multimg_sft_with_depth.json"
DEPTH_MAP="${DEPTH_OUTPUT_ROOT}/depth_map.json"
ENCODER="vitl"  # Depth Anything V2 encoder: vits/vitb/vitl/vitg
CHECKPOINT="/local_data/cy2932/checkpoints/depth/depth_anything_v2_vitl.pth"  # checkpoint路径

# ====== 步骤1: 生成depth图像 ======
echo "=========================================="
echo "步骤1: 生成depth图像"
echo "=========================================="
python tools/gen_official_depth.py \
    --input-root "${FIVE_FRAMES_ROOT}" \
    --output-root "${DEPTH_OUTPUT_ROOT}" \
    --encoder "${ENCODER}" \
    ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"}

if [ $? -ne 0 ]; then
    echo "❌ Depth生成失败！"
    exit 1
fi

echo "✅ Depth生成完成"
echo "   输出目录: ${DEPTH_OUTPUT_ROOT}"
echo "   映射文件: ${DEPTH_MAP}"

# ====== 步骤2: 构建SFT训练数据（带depth） ======
echo ""
echo "=========================================="
echo "步骤2: 构建SFT训练数据（带depth）"
echo "=========================================="
python tools/build_crossview_multimg_sft.py \
    --five_frames_root "${FIVE_FRAMES_ROOT}" \
    --out_json "${SFT_JSON}" \
    --depth_root "${DEPTH_OUTPUT_ROOT}" \
    --depth_map "${DEPTH_MAP}" \
    --neg_ratio 0.15 \
    --neg_tiers "40,40,20" \
    --max_pairs_per_uid 8

if [ $? -ne 0 ]; then
    echo "❌ SFT数据构建失败！"
    exit 1
fi

echo "✅ SFT数据构建完成"
echo "   输出文件: ${SFT_JSON}"

# ====== 步骤3: 验证depth pipeline ======
echo ""
echo "=========================================="
echo "步骤3: 验证depth pipeline"
echo "=========================================="
python tools/check_depth_pipeline.py \
    --sft_json "${SFT_JSON}" \
    --image_root "${FIVE_FRAMES_ROOT}" \
    --depth_root "${DEPTH_OUTPUT_ROOT}" \
    --num_samples 5 \
    --visualize

if [ $? -ne 0 ]; then
    echo "⚠️  Depth pipeline验证发现问题，但继续训练..."
fi

# ====== 步骤4: 训练200步（验证depth传递） ======
echo ""
echo "=========================================="
echo "步骤4: 训练200步（验证depth传递）"
echo "=========================================="

# 注意：这里需要先注册数据集
# 假设数据集已注册为 crossview_multimg_with_depth

OUTPUT_DIR="runs/train/RoboRefer-2B-CrossView-MultiImg-WithDepth-Debug"
mkdir -p "${OUTPUT_DIR}/model"

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "./runs/train/RoboRefer-2B-Depth-Align" \
    --chat_template qwen2 \
    --data_mixture "crossview_multimg_with_depth" \
    --vision_tower "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448" \
    --depth_tower "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448" \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample_3x3_fix \
    --depth_projector mlp_downsample_3x3_fix \
    --enable_depth True \
    --use_depth_tower True \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --tune_depth_tower True \
    --tune_depth_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}/model" \
    --num_train_epochs 1 \
    --max_steps 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none

echo ""
echo "=========================================="
echo "✅ 完整pipeline执行完成！"
echo "=========================================="
echo "检查训练日志中的 [DEBUG] 信息，确认depth tensor已正确传递"
