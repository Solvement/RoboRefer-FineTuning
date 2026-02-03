#!/bin/bash
# 生成拼接图格式的跨视角一致性训练数据（只包含正例，不包含负例）

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
    OUTPUT_PREFIX="crossview_concat_corrected_original"
    echo "📂 使用原始数据（未降采样）: $FIVE_FRAMES_ROOT"
elif [ "$DATA_VERSION" == "x3" ]; then
    FIVE_FRAMES_ROOT="$FIVE_FRAMES_ROOT_X3"
    OUTPUT_PREFIX="crossview_concat_corrected_x3"
    echo "📂 使用降采样数据（680x440）: $FIVE_FRAMES_ROOT"
else
    echo "❌ 错误: 数据版本必须是 'original' 或 'x3'"
    echo "用法: $0 [original|x3]"
    exit 1
fi

OUTPUT_DIR="tmp/${OUTPUT_PREFIX}"
CONCAT_ROOT="${OUTPUT_DIR}/images"
OUTPUT_JSON="${OUTPUT_DIR}/crossview_concat_sft.json"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CONCAT_ROOT"

echo "=========================================="
echo "生成拼接图格式的跨视角一致性训练数据"
echo "=========================================="
echo "数据版本: $DATA_VERSION"
echo "数据路径: $FIVE_FRAMES_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "拼接图目录: $CONCAT_ROOT"
echo ""
echo "⚠️  注意：只生成正例，不包含负例"
echo ""

# 检查数据目录是否存在
if [ ! -d "$FIVE_FRAMES_ROOT" ]; then
    echo "❌ 数据目录不存在: $FIVE_FRAMES_ROOT"
    exit 1
fi

echo "=========================================="
echo "开始生成数据..."
echo "=========================================="

python tools/build_crossview_concat_sft.py \
    --five_frames_root "$FIVE_FRAMES_ROOT" \
    --out_json "$OUTPUT_JSON" \
    --concat_root "$CONCAT_ROOT" \
    --max_pairs_per_uid 8 \
    --min_mask_area 100 \
    --split both

if [ $? -ne 0 ]; then
    echo "❌ 数据生成失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 数据生成完成！"
echo "=========================================="
echo "JSON文件: $OUTPUT_JSON"
echo "拼接图目录: $CONCAT_ROOT"
echo ""
echo "下一步: 开始训练"
echo "  export DATA_VERSION=$DATA_VERSION"
echo "  python run_crossview_training_concat.py"
