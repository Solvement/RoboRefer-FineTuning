#!/usr/bin/env python3
"""
Split crossview multi-image SFT JSON by neg_type for curriculum training.

Input (default):
  tmp/crossview_multimg_sft_25pct_with_depth.json

Outputs:
  tmp/crossview_multimg_sft_25pct_pos_only.json      (meta.is_neg == False)
  tmp/crossview_multimg_sft_25pct_pos_tierA.json     (pos + tierA)
  tmp/crossview_multimg_sft_25pct_all.json           (all samples, copy of filtered input)

All fields (image, depth, conversations, meta, etc.) are preserved.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return json.load(f)


def save_json(data: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def split_by_neg_type(src: Path, out_pos: Path, out_pos_tierA: Path, out_all: Path) -> None:
    data = load_json(src)
    pos_only: List[Dict] = []
    pos_and_tierA: List[Dict] = []
    all_samples: List[Dict] = []

    for s in data:
        meta = s.get("meta", {})
        is_neg = bool(meta.get("is_neg", False))
        neg_type = meta.get("neg_type", None)

        # all samples: keep everything (for explicit curriculum stage3)
        all_samples.append(s)

        if not is_neg:
            pos_only.append(s)
            pos_and_tierA.append(s)
        else:
            if neg_type == "tierA":
                pos_and_tierA.append(s)

    print(f"Total samples: {len(data)}")
    print(f"  POS only:     {len(pos_only)}")
    print(f"  POS+TierA:    {len(pos_and_tierA)}")
    print(f"  ALL (copy):   {len(all_samples)}")

    save_json(pos_only, out_pos)
    save_json(pos_and_tierA, out_pos_tierA)
    save_json(all_samples, out_all)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        default="tmp/crossview_multimg_sft_25pct_with_depth.json",
        help="Source SFT JSON (25% multi-image + depth).",
    )
    parser.add_argument(
        "--out-pos",
        type=str,
        default="tmp/crossview_multimg_sft_25pct_pos_only.json",
        help="Output JSON: positives only.",
    )
    parser.add_argument(
        "--out-pos-tierA",
        type=str,
        default="tmp/crossview_multimg_sft_25pct_pos_tierA.json",
        help="Output JSON: positives + tierA negatives.",
    )
    parser.add_argument(
        "--out-all",
        type=str,
        default="tmp/crossview_multimg_sft_25pct_all.json",
        help="Output JSON: all samples (copy).",
    )
    args = parser.parse_args()

    split_by_neg_type(Path(args.src), Path(args.out_pos), Path(args.out_pos_tierA), Path(args.out_all))


if __name__ == "__main__":
    main()

