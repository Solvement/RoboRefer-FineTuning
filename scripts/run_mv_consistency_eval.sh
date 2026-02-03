#!/bin/bash
# 多视角一致性评估脚本

set -e

REPO_ROOT="/local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer"
cd "$REPO_ROOT"

echo "=========================================="
echo "多视角一致性评估"
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

# 2. 运行评估
echo ""
echo "步骤2: 运行评估..."
"$PYTHON" tools/eval_mv_consistency.py \
  --model_name_or_path ./runs/train/mv_consistency_sft_v2 \
  --data_path ./tmp/mv_consistency_sft_v2/mv_val.json \
  --image_folder / \
  --chat_template qwen2 \
  --image_aspect_ratio dynamic \
  --max_new_tokens 64 \
  --output_json ./runs/train/mv_consistency_sft_v2/eval_mv_val_preds.json

if [ $? -ne 0 ]; then
    echo "❌ 评估失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 评估完成！"
echo "=========================================="
