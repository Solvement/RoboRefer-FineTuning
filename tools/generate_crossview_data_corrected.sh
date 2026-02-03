#!/bin/bash
# 生成跨视角一致性训练数据（支持原始数据和降采样数据）

# ====== 配置 ======
# 选择数据路径：
# - 原始数据（未降采样）
FIVE_FRAMES_ROOT_ORIGINAL="/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected"
# - 降采样数据（680x440，降采样3倍）
FIVE_FRAMES_ROOT_X3="/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3"

# 选择使用哪个数据路径（original 或 x3）
DATA_VERSION="${1:-x3}"  # 默认使用x3版本

if [ "$DATA_VERSION" == "original" ]; then
    FIVE_FRAMES_ROOT="$FIVE_FRAMES_ROOT_ORIGINAL"
    OUTPUT_PREFIX="crossview_corrected_original"
    echo "📂 使用原始数据（未降采样）: $FIVE_FRAMES_ROOT"
elif [ "$DATA_VERSION" == "x3" ]; then
    FIVE_FRAMES_ROOT="$FIVE_FRAMES_ROOT_X3"
    OUTPUT_PREFIX="crossview_corrected_x3"
    echo "📂 使用降采样数据（680x440）: $FIVE_FRAMES_ROOT"
else
    echo "❌ 错误: 数据版本必须是 'original' 或 'x3'"
    echo "用法: $0 [original|x3]"
    exit 1
fi

# Depth数据路径（可选）
DEPTH_ROOT="${2:-}"  # 如果提供第二个参数，作为depth路径
OUTPUT_DIR="tmp/${OUTPUT_PREFIX}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "生成跨视角一致性训练数据"
echo "=========================================="
echo "数据版本: $DATA_VERSION"
echo "数据路径: $FIVE_FRAMES_ROOT"
echo "输出目录: $OUTPUT_DIR"

# 检查数据目录是否存在
if [ ! -d "$FIVE_FRAMES_ROOT" ]; then
    echo "❌ 数据目录不存在: $FIVE_FRAMES_ROOT"
    exit 1
fi

# 检查depth数据是否存在
if [ -n "$DEPTH_ROOT" ] && [ -d "$DEPTH_ROOT" ]; then
    echo "✅ 找到Depth数据目录: $DEPTH_ROOT"
    DEPTH_ARG="--depth_root $DEPTH_ROOT"
else
    echo "⚠️  未提供Depth数据，将生成不带depth的数据"
    DEPTH_ARG=""
fi

echo ""
echo "=========================================="
echo "生成大规模训练数据（推荐）"
echo "=========================================="
echo "参数配置:"
echo "  - 负例比例: 15%"
echo "  - Tier分布: 20,30,50 (降低Tier A，提升Tier C)"
echo "  - 每个uid最多生成: 8对"
echo ""

python tools/build_crossview_multimg_sft.py \
    --five_frames_root "$FIVE_FRAMES_ROOT" \
    --out_json "$OUTPUT_DIR/crossview_multimg_sft_large.json" \
    --neg_ratio 0.15 \
    --neg_tiers "20,30,50" \
    --max_pairs_per_uid 8 \
    $DEPTH_ARG

if [ $? -ne 0 ]; then
    echo "❌ 数据生成失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "生成小规模训练数据（用于快速测试）"
echo "=========================================="
echo "参数配置:"
echo "  - 负例比例: 15%"
echo "  - Tier分布: 20,30,50"
echo "  - 每个uid最多生成: 4对"
echo ""

python tools/build_crossview_multimg_sft.py \
    --five_frames_root "$FIVE_FRAMES_ROOT" \
    --out_json "$OUTPUT_DIR/crossview_multimg_sft_small.json" \
    --neg_ratio 0.15 \
    --neg_tiers "20,30,50" \
    --max_pairs_per_uid 4 \
    $DEPTH_ARG

if [ $? -ne 0 ]; then
    echo "❌ 小规模数据生成失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 数据生成完成！"
echo "=========================================="
echo "大规模数据: $OUTPUT_DIR/crossview_multimg_sft_large.json"
echo "小规模数据: $OUTPUT_DIR/crossview_multimg_sft_small.json"
echo ""
echo "下一步: 更新数据集配置并开始训练"
echo "  1. 运行此脚本生成数据: bash tools/generate_crossview_data_corrected.sh [original|x3]"
echo "  2. 更新 llava/data/datasets_mixture.py 中的数据集配置"
echo "  3. 运行训练脚本: python run_crossview_training_corrected.py"
