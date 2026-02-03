#!/usr/bin/env python3
"""
Generic evaluation script for spatial tasks (Task1 / Task2A / Task2B).

Supports:
- Multi-image format: sample["image"] is a list of image paths (e.g. [A, B]).
- Concat format: sample["image"] is a single concatenated image path.

Behavior:
- Inserts <image> tokens according to number of images:
  * multi-image: num_images = len(image_list)
  * concat: num_images = 1
- Prompts come from sample["conversations"][0]["value"] and MUST NOT contain <image>.
- GT comes from sample["conversations"][1]["value"]:
  * "NOT_VISIBLE" or
  * "[(x, y)]" with x, y in [0,1]

Metrics:
- NOT_VISIBLE accuracy (on samples whose GT is NOT_VISIBLE).
- For visible GT:
  * L2 distance between predicted and GT point.
  * Success@0.02 / 0.05 / 0.10 wrt L2.
  * Hit@Mask: whether predicted point falls inside the GT mask.

Mask lookup:
- Prefer explicit "query_original" field in sample if present.
- Else, if multi-image with len(image) >= 2, use image[-1] as query.
- Mask path is inferred as:
  query_path.replace("_original.png", "_mask_dialated.png") or "_mask.png"
  (and similarly for inpainted case: we still use the same suffix rule).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_image
from llava.model import LlavaLlamaModel


NOT_VISIBLE_TOKEN = "NOT_VISIBLE"
COORD_PATTERN = re.compile(r"\[\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\]")


def parse_prediction(text: str) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """Parse model prediction -> (is_not_visible, (x,y) or None)."""
    upper = text.upper()
    if NOT_VISIBLE_TOKEN in upper:
        return True, None
    m = COORD_PATTERN.search(text)
    if m:
        try:
            x = float(m.group(1))
            y = float(m.group(2))
            return False, (x, y)
        except Exception:
            return False, None
    return False, None


def parse_gt(text: str) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """Parse GT string."""
    upper = text.upper()
    if upper == NOT_VISIBLE_TOKEN:
        return True, None
    m = COORD_PATTERN.search(text)
    if m:
        try:
            x = float(m.group(1))
            y = float(m.group(2))
            return False, (x, y)
        except Exception:
            return False, None
    return False, None


def compute_l2(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load LlavaLlamaModel + tokenizer, similar to eval_mv_consistency."""
    from transformers import AutoConfig

    mpath = Path(model_path)
    if (mpath / "model").exists():
        mpath = mpath / "model"

    config = AutoConfig.from_pretrained(str(mpath), trust_remote_code=True)
    config.resume_path = str(mpath)

    model = LlavaLlamaModel.from_pretrained(
        str(mpath),
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = model.tokenizer

    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        chat_template_path = (
            Path(__file__).parent.parent
            / "llava"
            / "model"
            / "language_model"
            / "chat_templates"
            / "qwen2.jinja"
        )
        if chat_template_path.exists():
            with open(chat_template_path) as f:
                chat_template = f.read().replace("    ", "").replace("\n", "")
            tokenizer.chat_template = chat_template

    return model, tokenizer


def build_prompt_with_images(human_prompt: str, num_images: int) -> str:
    """Insert <image> tokens BEFORE the human prompt (mimic preprocess_rgbd)."""
    tokens = f"{DEFAULT_IMAGE_TOKEN}\n" * num_images
    return tokens + human_prompt


def format_conversation(prompt: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": None},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def infer_query_image_path(sample: Dict[str, Any]) -> Optional[str]:
    """
    Return the path whose mask we should evaluate against.
    Priority:
    - sample["query_original"] if present
    - elif list image and len>=2: image[-1]
    - else: None (no mask-based metric).
    """
    if "query_original" in sample:
        return sample["query_original"]
    img_field = sample.get("image", None)
    if isinstance(img_field, list) and len(img_field) >= 2:
        return img_field[-1]
    return None


def infer_mask_path(query_path: Path) -> Optional[Path]:
    """
    Infer mask path from query image path.
    - Prefer *_mask_dialated.png
    - Fallback *_mask.png
    For inpainted views we still assume suffix scheme is consistent.
    """
    stem = query_path.name
    # Very simple replacement rules; we only touch exact suffixes.
    candidates = []
    if stem.endswith("_original.png"):
        candidates.append(query_path.with_name(stem.replace("_original.png", "_mask_dialated.png")))
        candidates.append(query_path.with_name(stem.replace("_original.png", "_mask.png")))
    elif stem.endswith("_inpainted.png"):
        candidates.append(query_path.with_name(stem.replace("_inpainted.png", "_mask_dialated.png")))
        candidates.append(query_path.with_name(stem.replace("_inpainted.png", "_mask.png")))
    else:
        # Try generic
        candidates.append(query_path.with_name(stem.replace(".png", "_mask_dialated.png")))
        candidates.append(query_path.with_name(stem.replace(".png", "_mask.png")))

    for c in candidates:
        if c.exists():
            return c
    return None


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return arr


def hit_at_mask(
    pred_coord: Tuple[float, float],
    mask: np.ndarray,
) -> bool:
    """Check whether predicted point lies inside mask>0 in the mask's own coordinate system.

    `pred_coord` is normalized (x,y) in [0,1] **wrt the same image as the mask**.
    For concat Task1, caller MUST first convert (x_full,y_full) from full-concat
    coordinates into local right-half coordinates before calling this.
    """
    h, w = mask.shape
    x_norm, y_norm = pred_coord
    x = int(round(x_norm * (w - 1)))
    y = int(round(y_norm * (h - 1)))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return bool(mask[y, x] > 0)


def run_inference_on_sample(
    model: LlavaLlamaModel,
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    image_folder: str,
    device: str,
    max_new_tokens: int,
) -> str:
    """Run model.generate on one sample."""
    human_prompt = sample["conversations"][0]["value"]
    img_field = sample.get("image", None)

    if isinstance(img_field, list):
        image_paths = img_field
    elif isinstance(img_field, str):
        image_paths = [img_field]
    else:
        raise ValueError("sample['image'] must be list or string.")

    num_images = len(image_paths)
    prompt_with_images = build_prompt_with_images(human_prompt, num_images=num_images)
    prompt_str = format_conversation(prompt_with_images, tokenizer)

    # Load images (absolute paths; image_folder is usually "/").
    images = []
    for p in image_paths:
        img = process_image(p, None, image_folder)
        images.append(img)
    if len(images) == 1:
        images_tensor = images[0].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,H,W)
    else:
        images_tensor = torch.stack(images).unsqueeze(0).to(device)  # (1,N,C,H,W)

    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    with torch.no_grad():
        out = model.generate(
            **inputs,
            images=images_tensor,
            depths=None,
            **gen_kwargs,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if prompt_str in text:
        text = text.split(prompt_str, 1)[-1]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate spatial tasks (Task1 / Task2A / Task2B).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="/")
    parser.add_argument("--chat_template", type=str, default="qwen2")
    parser.add_argument(
        "--image_aspect_ratio",
        type=str,
        default="resize",
        help="Kept for compatibility; actual resizing is handled inside model's image processor.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    data_path = Path(args.data_json)
    with open(data_path, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if args.limit is not None:
        data = data[: args.limit]

    print("===========================================")
    print("Spatial tasks evaluation")
    print("===========================================")
    print(f" model       : {args.model_name_or_path}")
    print(f" data_json   : {data_path}")
    print(f" num_samples : {len(data)}")
    print(f" output_json : {args.output_json}")
    print()

    print("Loading model ...")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device=args.device)
    print("✅ Model loaded.")
    print()

    stats = {
        "total": 0,
        "gt_not_visible": 0,
        "pred_not_visible": 0,
        "correct_not_visible": 0,
        "visible": 0,
        "l2_errors": [],
        "success_002": 0,
        "success_005": 0,
        "success_010": 0,
        "hit_mask_count": 0,
        "hit_mask_total": 0,
    }
    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(data):
        if (idx + 1) % 50 == 0:
            print(f"  progress: {idx+1}/{len(data)}")

        sid = sample.get("id", f"sample_{idx}")
        gt_text = sample["conversations"][1]["value"]
        gt_is_nv, gt_coord = parse_gt(gt_text)

        try:
            pred_text = run_inference_on_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                image_folder=args.image_folder,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            pred_is_nv, pred_coord = parse_prediction(pred_text)
        except Exception as e:
            print(f"⚠️  sample {sid} failed: {e}")
            pred_text = ""
            pred_is_nv, pred_coord = False, None

        stats["total"] += 1
        if pred_is_nv:
            stats["pred_not_visible"] += 1

        rec: Dict[str, Any] = {
            "id": sid,
            "pred_text": pred_text,
            "pred_parsed": {"is_not_visible": pred_is_nv, "coord": pred_coord},
            "gt": {"text": gt_text, "is_not_visible": gt_is_nv, "coord": gt_coord},
            "l2_error": None,
            "success_l2_010": False,
            "hit_mask": None,
        }

        if gt_is_nv:
            stats["gt_not_visible"] += 1
            if pred_is_nv:
                stats["correct_not_visible"] += 1
                rec["success_l2_010"] = True  # treat correct NOT_VISIBLE as success
        else:
            stats["visible"] += 1
            if (not pred_is_nv) and (pred_coord is not None) and (gt_coord is not None):
                l2 = compute_l2(pred_coord, gt_coord)
                stats["l2_errors"].append(l2)
                rec["l2_error"] = l2
                if l2 <= 0.02:
                    stats["success_002"] += 1
                if l2 <= 0.05:
                    stats["success_005"] += 1
                if l2 <= 0.10:
                    stats["success_010"] += 1
                    rec["success_l2_010"] = True

            # Hit@Mask: only meaningful if we have a mask and a predicted point.
            # NOTE: for concat Task1 the model outputs coords in FULL-concat space.
            # We must map (x_full,y_full) back to the query image coords before checking the mask.
            q_path_str = infer_query_image_path(sample)
            if q_path_str is not None and pred_coord is not None and not pred_is_nv:
                q_path = Path(q_path_str)
                mask_path = infer_mask_path(q_path)
                if mask_path is not None:
                    mask = load_mask(mask_path)
                    if mask is not None:
                        stats["hit_mask_total"] += 1

                        # Determine whether this is a concat sample (single image path).
                        is_concat = isinstance(sample.get("image", None), str)
                        if is_concat:
                            # concat: (x_full,y_full) normalized over full concat.
                            x_full, y_full = pred_coord
                            if x_full < 0.5:
                                # Point lies on left half (reference) → always miss for query mask.
                                hit = False
                            else:
                                # Map to right-half local coords.
                                xR = (x_full - 0.5) * 2.0
                                yR = y_full
                                xR = max(0.0, min(1.0, xR))
                                yR = max(0.0, min(1.0, yR))
                                hit = hit_at_mask((xR, yR), mask)
                        else:
                            # Multi-image: prediction already wrt query image's own frame.
                            hit = hit_at_mask(pred_coord, mask)

                        if hit:
                            stats["hit_mask_count"] += 1
                        rec["hit_mask"] = bool(hit)

        results.append(rec)

    # Aggregate metrics
    not_visible_acc = (
        stats["correct_not_visible"] / stats["gt_not_visible"] if stats["gt_not_visible"] > 0 else 0.0
    )
    if stats["l2_errors"]:
        l2_arr = np.array(stats["l2_errors"], dtype=np.float32)
        mean_l2 = float(l2_arr.mean())
        std_l2 = float(l2_arr.std())
        median_l2 = float(np.median(l2_arr))
    else:
        mean_l2 = std_l2 = median_l2 = None

    success_002 = stats["success_002"] / stats["visible"] if stats["visible"] > 0 else 0.0
    success_005 = stats["success_005"] / stats["visible"] if stats["visible"] > 0 else 0.0
    success_010 = stats["success_010"] / stats["visible"] if stats["visible"] > 0 else 0.0
    hit_mask = stats["hit_mask_count"] / stats["hit_mask_total"] if stats["hit_mask_total"] > 0 else 0.0

    print()
    print("===========================================")
    print("Results")
    print("===========================================")
    print(f"Total samples        : {stats['total']}")
    print(f"GT NOT_VISIBLE       : {stats['gt_not_visible']}")
    print(f"Visible GT           : {stats['visible']}")
    print()
    print(f"NOT_VISIBLE accuracy : {not_visible_acc:.4f} "
          f"({stats['correct_not_visible']}/{stats['gt_not_visible']})")
    if mean_l2 is not None:
        print()
        print("L2 error (visible GT only):")
        print(f"  mean   = {mean_l2:.4f}")
        print(f"  std    = {std_l2:.4f}")
        print(f"  median = {median_l2:.4f}")
    print()
    print(f"Success@0.02         : {success_002:.4f} ({stats['success_002']}/{stats['visible']})")
    print(f"Success@0.05         : {success_005:.4f} ({stats['success_005']}/{stats['visible']})")
    print(f"Success@0.10         : {success_010:.4f} ({stats['success_010']}/{stats['visible']})")
    print(f"Hit@Mask             : {hit_mask:.4f} ({stats['hit_mask_count']}/{stats['hit_mask_total']})")
    print("===========================================")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {
            "total": stats["total"],
            "gt_not_visible": stats["gt_not_visible"],
            "visible_gt": stats["visible"],
            "not_visible_accuracy": float(not_visible_acc),
            "mean_l2_error": mean_l2,
            "std_l2_error": std_l2,
            "median_l2_error": median_l2,
            "success_002": float(success_002),
            "success_005": float(success_005),
            "success_010": float(success_010),
            "hit_at_mask": float(hit_mask),
            "hit_at_mask_count": stats["hit_mask_count"],
            "hit_at_mask_total": stats["hit_mask_total"],
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved detailed results to: {out_path}")


if __name__ == "__main__":
    main()

