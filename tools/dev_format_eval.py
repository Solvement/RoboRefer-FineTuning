#!/usr/bin/env python3
"""
Dev-format evaluation for crossview multi-image SFT.

- Builds a small dev set: 20 positives + 20 negatives from the original 25% JSON.
- Runs greedy inference (temperature=0, max_new_tokens=16).
- Reports:
    * POS:  %valid_coord_format, %NOT_VISIBLE
    * NEG:  %NOT_VISIBLE
    * Coord mean / std over all valid coords

This script uses LlavaLlamaModel + mm_utils; it does NOT modify training logic.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from llava import conversation as conversation_lib
from llava.mm_utils import process_depth, process_image
from llava.model import LlavaLlamaModel


NOT_VISIBLE_TOKEN = "NOT_VISIBLE"
COORD_PATTERN = re.compile(r"\\[\\(\\s*([0-9.]+)\\s*,\\s*([0-9.]+)\\s*\\)\\]")


def load_json(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return json.load(f)


def select_dev_indices(data: List[Dict], n_pos: int = 20, n_neg: int = 20) -> Tuple[List[int], List[int]]:
    pos_idx, neg_idx = [], []
    for i, s in enumerate(data):
        meta = s.get("meta", {})
        is_neg = bool(meta.get("is_neg", False))
        if not is_neg and len(pos_idx) < n_pos:
            pos_idx.append(i)
        elif is_neg and len(neg_idx) < n_neg:
            neg_idx.append(i)
        if len(pos_idx) >= n_pos and len(neg_idx) >= n_neg:
            break
    return pos_idx, neg_idx


def build_prompt(sample: Dict) -> str:
    convs = sample.get("conversations", [])
    if not convs:
        return ""
    # assume first turn is human prompt
    return convs[0]["value"]


def run_inference(
    model: LlavaLlamaModel,
    tokenizer: AutoTokenizer,
    sample: Dict,
    image_root: Path,
    depth_root: Path,
    device: str = "cuda",
    max_new_tokens: int = 16,
) -> str:
    model.eval()
    with torch.no_grad():
        prompt = build_prompt(sample)
        conv = conversation_lib.conv_templates["qwen2"].copy()
        conv.append_message("human", prompt)
        conv.append_message("gpt", None)
        prompt_str = conv.get_prompt()

        image_paths = sample.get("image", [])
        depth_paths = sample.get("depth", []) or []

        images = []
        depths = []
        for p in image_paths:
            images.append(process_image(p, None, str(image_root)))
        for d in depth_paths:
            depths.append(process_depth(d, None, str(depth_root)))

        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

        out = model.generate(
            **inputs,
            images=torch.stack(images).unsqueeze(0) if images else None,
            depths=torch.stack(depths).unsqueeze(0) if depths else None,
            **gen_kwargs,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if prompt_str in text:
            text = text.split(prompt_str, 1)[-1]
        return text.strip()


def analyze_output(text: str) -> Tuple[bool, bool, List[float]]:
    """Return (is_not_visible, has_valid_coord, coords_list)."""
    is_nv = NOT_VISIBLE_TOKEN in text
    m = COORD_PATTERN.search(text)
    coords: List[float] = []
    has_valid = False
    if m:
        has_valid = True
        x, y = float(m.group(1)), float(m.group(2))
        coords = [x, y]
    return is_nv, has_valid, coords


def eval_dev(
    model_path: Path,
    sft_json: Path,
    image_root: Path,
    depth_root: Path,
    device: str = "cuda",
) -> Dict:
    data = load_json(sft_json)
    pos_idx, neg_idx = select_dev_indices(data)
    print(f"Dev set: {len(pos_idx)} pos, {len(neg_idx)} neg")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = LlavaLlamaModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16).to(device)

    stats: Dict[str, Dict] = {
        "pos": {"n": 0, "nv": 0, "valid": 0, "coords": []},
        "neg": {"n": 0, "nv": 0, "valid": 0, "coords": []},
    }

    def update(split: str, is_nv: bool, has_valid: bool, coords: List[float]) -> None:
        s = stats[split]
        s["n"] += 1
        if is_nv:
            s["nv"] += 1
        if has_valid:
            s["valid"] += 1
            s["coords"].extend(coords)

    for i in pos_idx:
        text = run_inference(model, tokenizer, data[i], image_root, depth_root, device=device)
        is_nv, has_valid, coords = analyze_output(text)
        update("pos", is_nv, has_valid, coords)

    for i in neg_idx:
        text = run_inference(model, tokenizer, data[i], image_root, depth_root, device=device)
        is_nv, has_valid, coords = analyze_output(text)
        update("neg", is_nv, has_valid, coords)

    metrics: Dict[str, Dict] = {}
    for split in ["pos", "neg"]:
        s = stats[split]
        n = max(s["n"], 1)
        pct_nv = s["nv"] / n * 100.0
        pct_valid = s["valid"] / n * 100.0
        coord_output_rate = 100.0 - pct_nv  # 1 - NOT_VISIBLE_rate

        # collapse check: round coords to 2 decimals, then count uniques
        rounded_coords = []
        if s["coords"]:
            arr = np.array(s["coords"])
            mean = float(arr.mean())
            std = float(arr.std())
            # pair-wise (x,y)
            for i in range(0, len(s["coords"]), 2):
                if i + 1 < len(s["coords"]):
                    x = round(float(s["coords"][i]), 2)
                    y = round(float(s["coords"][i + 1]), 2)
                    rounded_coords.append((x, y))
            if rounded_coords:
                uniq, counts = np.unique(rounded_coords, axis=0, return_counts=True)
                unique_ratio = len(uniq) / max(len(rounded_coords), 1)
                top1_freq = int(counts.max())
            else:
                unique_ratio = float("nan")
                top1_freq = 0
        else:
            mean = float("nan")
            std = float("nan")
            unique_ratio = float("nan")
            top1_freq = 0

        metrics[split] = {
            "n": s["n"],
            "pct_NOT_VISIBLE": pct_nv,
            "pct_valid_coord": pct_valid,
            "coord_output_rate": coord_output_rate,
            "coord_mean": mean,
            "coord_std": std,
            "unique_ratio_2dec": unique_ratio,
            "top1_freq_2dec": top1_freq,
        }

        print(
            f"[{split}] n={s['n']}, %NOT_VISIBLE={pct_nv:.1f}, "
            f"%valid_coord={pct_valid:.1f}, coord_output_rate={coord_output_rate:.1f}, "
            f"coord_mean={mean:.3f}, coord_std={std:.3f}, "
            f"unique_ratio_2dec={unique_ratio:.3f}, top1_freq_2dec={top1_freq}"
        )

    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to trained model (stage checkpoint).")
    p.add_argument(
        "--sft-json",
        type=str,
        default="tmp/crossview_multimg_sft_25pct_with_depth.json",
        help="Original 25%% SFT JSON (multi-image + depth).",
    )
    p.add_argument(
        "--image-root",
        type=str,
        default="/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3",
    )
    p.add_argument("--depth-root", type=str, default="tmp/scannet_depth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--jsonl-out",
        type=str,
        default=None,
        help="Optional path to append metrics as a single JSON line.",
    )
    args = p.parse_args()

    metrics = eval_dev(
        Path(args.model),
        Path(args.sft_json),
        Path(args.image_root),
        Path(args.depth_root),
        device=args.device,
    )

    if args.jsonl_out:
        out_path = Path(args.jsonl_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    main()

