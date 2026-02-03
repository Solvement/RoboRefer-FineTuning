#!/bin/bash
# 生成Curriculum Learning两阶段训练数据

FIVE_FRAMES_ROOT="/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3"
DEPTH_ROOT="/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3_depth"  # Depth Anything生成的depth数据
OUTPUT_DIR="tmp/curriculum"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "生成Curriculum Learning训练数据（带Depth）"
echo "=========================================="

# 检查depth数据是否存在
if [ ! -d "$DEPTH_ROOT" ]; then
    echo "⚠️  Depth数据目录不存在: $DEPTH_ROOT"
    echo "   将生成不带depth的数据"
    DEPTH_ARG=""
else
    echo "✅ 找到Depth数据目录: $DEPTH_ROOT"
    DEPTH_ARG="--depth_root $DEPTH_ROOT"
fi

# Phase 1: 0%负例，只用top 50%大目标
echo ""
echo "📚 Phase 1: 生成0%负例数据（只用top 50%大目标）..."
python tools/build_crossview_multimg_sft.py \
    --five_frames_root "$FIVE_FRAMES_ROOT" \
    --out_json "$OUTPUT_DIR/crossview_multimg_phase1.json" \
    --neg_ratio 0.0 \
    --neg_tiers "40,40,20" \
    --max_pairs_per_uid 8 \
    --curriculum_phase phase1 \
    --filter_top_percentile 0.5 \
    $DEPTH_ARG

# Phase 2: 15%负例，使用全部正例，调整tier分布
echo ""
echo "📚 Phase 2: 生成15%负例数据（使用全部正例，调整tier分布）..."
python tools/build_crossview_multimg_sft.py \
    --five_frames_root "$FIVE_FRAMES_ROOT" \
    --out_json "$OUTPUT_DIR/crossview_multimg_phase2.json" \
    --neg_ratio 0.15 \
    --neg_tiers "40,40,20" \
    --max_pairs_per_uid 8 \
    --curriculum_phase phase2 \
    $DEPTH_ARG

echo ""
echo "✅ 数据生成完成！"
echo "   Phase 1: $OUTPUT_DIR/crossview_multimg_phase1.json"
echo "   Phase 2: $OUTPUT_DIR/crossview_multimg_phase2.json"
