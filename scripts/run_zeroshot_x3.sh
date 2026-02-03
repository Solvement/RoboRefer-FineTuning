#!/bin/bash
# Zero-shot evaluation on x3 spatial tasks (Task1 / Task2A / Task2B).

set -e

REPO_ROOT="/local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer"
cd "$REPO_ROOT"

echo "=========================================="
echo "Zero-shot evaluation on x3 spatial tasks"
echo "=========================================="

PYTHON="/local_data/ky2738/envs/snpp2msg-rast/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "❌ Python not found: $PYTHON"
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUT_DIR="./tmp/spatial_tasks_x3"
MODEL="./runs/train/RoboRefer-2B-Depth-Align"
EVAL_DIR="./runs/eval_spatial_x3"

mkdir -p "$OUT_DIR" "$EVAL_DIR"

echo ""
echo "Step 1: Build x3 spatial task datasets (if missing) ..."
if [ -f "$OUT_DIR/t1_val_multi.json" ] && [ -f "$OUT_DIR/t1_val_concat.json" ]; then
  echo "✅ Found existing spatial_tasks_x3 JSONs under $OUT_DIR, skipping build."
else
  "$PYTHON" tools/build_spatial_tasks_x3.py \
    --data_root /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3 \
    --out_dir "$OUT_DIR" \
    --mode anchor \
    --anchor_k 1 \
    --emit_concat 1
fi

echo ""
echo "Step 2: Run zero-shot eval for each task ..."

echo "  2a) Task1 (multi-image, val)"
"$PYTHON" tools/eval_spatial_tasks.py \
  --model_name_or_path "$MODEL" \
  --data_json "$OUT_DIR/t1_val_multi.json" \
  --image_folder / \
  --chat_template qwen2 \
  --image_aspect_ratio resize \
  --max_new_tokens 64 \
  --output_json "$EVAL_DIR/t1_val_multi.json"

echo ""
echo "  2b) Task1 (concat, val)"
"$PYTHON" tools/eval_spatial_tasks.py \
  --model_name_or_path "$MODEL" \
  --data_json "$OUT_DIR/t1_val_concat.json" \
  --image_folder / \
  --chat_template qwen2 \
  --image_aspect_ratio resize \
  --max_new_tokens 64 \
  --output_json "$EVAL_DIR/t1_val_concat.json"

echo ""
echo "  2c) Task2A (missing across view, val)"
"$PYTHON" tools/eval_spatial_tasks.py \
  --model_name_or_path "$MODEL" \
  --data_json "$OUT_DIR/t2a_val_multi.json" \
  --image_folder / \
  --chat_template qwen2 \
  --image_aspect_ratio resize \
  --max_new_tokens 64 \
  --output_json "$EVAL_DIR/t2a_val_multi.json"

echo ""
echo "  2d) Task2B (difference grounding, val)"
"$PYTHON" tools/eval_spatial_tasks.py \
  --model_name_or_path "$MODEL" \
  --data_json "$OUT_DIR/t2b_val_multi.json" \
  --image_folder / \
  --chat_template qwen2 \
  --image_aspect_ratio resize \
  --max_new_tokens 64 \
  --output_json "$EVAL_DIR/t2b_val_multi.json"

echo ""
echo "Step 3: Summary"

"$PYTHON" - << 'PY'
import json
from pathlib import Path

root = Path("runs/eval_spatial_x3")

def load_metrics(name):
    p = root / name
    if not p.exists():
        return name, None
    d = json.loads(p.read_text())
    return name, d.get("metrics", {})

names = [
    "t1_val_multi.json",
    "t1_val_concat.json",
    "t2a_val_multi.json",
    "t2b_val_multi.json",
]

print("==========================================")
print("Zero-shot x3 summary (key metrics)")
print("==========================================")
for n in names:
    name, m = load_metrics(n)
    print(f"\n[{name}]")
    if not m:
        print("  (no metrics)")
        continue
    def fmt(v):
        return "None" if v is None else f"{v:.4f}" if isinstance(v, (int,float)) else str(v)
    print(f"  Hit@Mask            : {fmt(m.get('hit_at_mask'))}")
    print(f"  NOT_VISIBLE acc     : {fmt(m.get('not_visible_accuracy'))}")
    print(f"  mean L2 (visible)   : {fmt(m.get('mean_l2_error'))}")
    print(f"  Success@0.02        : {fmt(m.get('success_002'))}")
    print(f"  Success@0.05        : {fmt(m.get('success_005'))}")
    print(f"  Success@0.10        : {fmt(m.get('success_010'))}")

print("\nDone.")
PY

echo ""
echo "=========================================="
echo "✅ Zero-shot x3 evaluation completed."
echo "=========================================="

