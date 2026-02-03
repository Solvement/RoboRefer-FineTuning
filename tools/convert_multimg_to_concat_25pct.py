#!/usr/bin/env python3
"""
从多图 CrossView SFT（25% 子集）生成拼接图版本：
- 输入: tmp/crossview_multimg_sft_25pct_with_depth.json
- 输出:
  - 拼接图根目录: ./tmp/crossview_concat_25pct_images
  - 拼接版 SFT:   ./tmp/crossview_concat_sft_25pct.json

说明：
- 只比较「多图 vs 拼接图」效果，这里拼接版暂不使用 depth（RGB-only），
  以避免改动太多模型结构；多图版本保持 RGB+Depth。
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def make_concat_image(img_a_path: Path, img_b_path: Path, out_path: Path):
    """横向拼接 A/B 两张图，并保存到 out_path。"""
    img_a = Image.open(img_a_path).convert("RGB")
    img_b = Image.open(img_b_path).convert("RGB")

    # 统一高度，按比例缩放宽度（保持纵横比）
    h = max(img_a.height, img_b.height)

    def resize_to_h(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img
        scale = target_h / img.height
        new_w = int(round(img.width * scale))
        return img.resize((new_w, target_h), Image.BILINEAR)

    img_a_r = resize_to_h(img_a, h)
    img_b_r = resize_to_h(img_b, h)

    w_total = img_a_r.width + img_b_r.width
    concat = Image.new("RGB", (w_total, h))
    concat.paste(img_a_r, (0, 0))
    concat.paste(img_b_r, (img_a_r.width, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat.save(out_path)


def convert_multimg_to_concat(
    src_json: Path,
    concat_root: Path,
    out_json: Path,
):
    data = json.load(open(src_json, "r"))

    print(f"📂 输入多图 SFT: {src_json} ({len(data)} samples)")
    print(f"📁 拼接图输出目录: {concat_root}")
    print(f"📝 拼接版 SFT 将写入: {out_json}")

    new_data = []

    for i, sample in enumerate(data):
        images = sample.get("image", [])
        if len(images) < 2:
            continue

        img_a = Path(images[0])
        img_b = Path(images[1])

        # 构造相对路径（沿用 scene/uid/filename 结构）
        # 输入是绝对路径：.../scannet_inpainted_dilate002_15obj_5frames_corrected_x3/train/scene/uid/file.png
        img_str = str(img_a)
        if "scannet_inpainted_dilate002_15obj_5frames_corrected_x3" in img_str:
            rel = img_str.split("scannet_inpainted_dilate002_15obj_5frames_corrected_x3/")[-1]
        else:
            # 回退：只用文件名
            rel = img_a.name

        rel_path = Path(rel)
        concat_rel = rel_path.with_name(rel_path.stem + "_concat.png")
        concat_full = concat_root / concat_rel

        try:
            make_concat_image(img_a, img_b, concat_full)
        except Exception as e:
            print(f"⚠️  拼接失败，跳过样本 {sample.get('id', i)}: {e}")
            continue

        # 构造新的 sample：image 列表只保留一张拼接图；去掉 depth 字段
        new_sample = dict(sample)
        new_sample["image"] = [str(concat_full)]
        new_sample.pop("depth", None)
        new_data.append(new_sample)

        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i+1} / {len(data)}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(new_data, open(out_json, "w"), indent=2)
    print(f"✅ 完成，输出样本数: {len(new_data)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="将多图 CrossView SFT (25%) 转换为拼接图版本")
    parser.add_argument(
        "--src-json",
        type=str,
        default="tmp/crossview_multimg_sft_25pct_with_depth.json",
        help="输入多图 SFT JSON（25%子集）",
    )
    parser.add_argument(
        "--concat-root",
        type=str,
        default="tmp/crossview_concat_25pct_images",
        help="拼接图输出根目录",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="tmp/crossview_concat_sft_25pct.json",
        help="输出拼接版 SFT JSON",
    )

    args = parser.parse_args()
    convert_multimg_to_concat(
        Path(args.src_json),
        Path(args.concat_root),
        Path(args.out_json),
    )


if __name__ == "__main__":
    main()

