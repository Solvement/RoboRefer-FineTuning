#!/usr/bin/env python3
"""
从 five_frames 标注构造 Cross-View 多图 SFT 训练数据（小规模/可控采样）

输入：
  - root_dir: /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3
    目录结构示例：
      root_dir/
        validation/
          f3d64c30f8/
            uid_129/
              f3d64c30f8_uid_129_five_frames.json
              01_008710_original.png
              01_008710_mask.png
              ...

five_frames JSON 结构示例（列表，长度通常为5）：
  {
    "scene_id": "f3d64c30f8",
    "uid": 129,
    "frame_id": "008710",
    "label": "bag",
    "img_w": 1920,
    "img_h": 1440,
    "x0": 11.0,
    "y0": 0.0,
    "x1": 598.0,
    "y1": 346.0,
    "cx_norm": 0.1585,
    "cy_norm": 0.1201,
    "original": ".../01_008710_original.png",
    "inpainted": ".../01_008710_inpainted.png",
    "mask": ".../01_008710_mask.png"
  }

输出：
  - SFT JSON，RoboRefer 格式，每条样本：
    {
      "id": "scene_uid_Aframe_Bframe",
      "image": [rel_path_A, rel_path_B],
      "conversations": [
        {"from": "human", "value": "<指令>"},
        {"from": "gpt", "value": "[(0.123, 0.456)] 或 NOT_VISIBLE"}
      ]
    }

说明：
  - 这里先实现一版“小规模测试用”的生成器，支持：
    - 仅从 validation 采样
    - 限制最多样本数（例如 20 / 50）
    - 正例：A,B 都来自同一 five_frames 且同一 uid
    - 负例（可选）：A 来自一个 uid，B 来自另一 scene/uid（easy negative）
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """读取mask为灰度图；读取失败返回None。"""
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return m


def sample_point_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    从mask中随机采样一个前景点，返回 (x_norm, y_norm)，归一化到[0,1]。
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    idx = np.random.randint(0, xs.size)
    y, x = int(ys[idx]), int(xs[idx])
    h, w = mask.shape[:2]
    if w <= 1 or h <= 1:
        return None
    return float(x) / float(w - 1), float(y) / float(h - 1)


def build_human_prompt(label: str) -> str:
    """
    构造简单版本的多图 cross-view 指令。
    注意：这里不写坐标数字，只说 A 标记物体，让模型在 B 里找同一个。
    """
    label_txt = label if label else "object"
    prompt = (
        "You are given TWO separate images:\n"
        "- Image A (REFERENCE): the target object is highlighted (marked) in the image.\n"
        "- Image B (QUERY): you need to find the SAME object as in Image A.\n\n"
        f"The target in Image A is a \"{label_txt}\". It is visually marked, so you can clearly see which object to track.\n\n"
        "TASK:\n"
        "1. Look at Image A and understand which object is marked.\n"
        "2. Look at Image B and determine whether the SAME object is visible.\n"
        "3. If the object is visible in Image B, output ONE point coordinate on that object.\n"
        "4. If the object is NOT visible in Image B, answer NOT_VISIBLE.\n\n"
        "OUTPUT FORMAT:\n"
        "- If visible: answer with one coordinate in normalized [0,1] range relative to Image B only, in the form: [(x, y)]\n"
        "- If NOT visible: answer exactly: NOT_VISIBLE\n"
    )
    return prompt


def build_sft_samples_from_five_frames(
    five_frames_path: Path,
    root_dir: Path,
    max_pairs_per_file: int = 4,
) -> List[Dict[str, Any]]:
    """
    给定一个 *_five_frames.json，构造若干 (A,B) 正例样本。

    策略（简单版本）：
      - 用第一个视角作为 A
      - 其余视角依次作为 B（最多 max_pairs_per_file 个）
      - B 上读取 mask，采样一个 GT 点（若mask无效则跳过该对）
    """
    data = json.loads(five_frames_path.read_text())
    if not isinstance(data, list) or len(data) < 2:
        return []

    # A 视角：第一个条目
    ref = data[0]
    ref_img_abs = Path(ref["original"])
    label = ref.get("label", "")

    # 将绝对路径转换为 root_dir 下的相对路径，便于 dataloader 使用统一前缀
    try:
        ref_img_rel = ref_img_abs.relative_to(root_dir)
    except ValueError:
        # 不在root_dir下时，直接用绝对路径（调试阶段可以接受）
        ref_img_rel = ref_img_abs

    samples: List[Dict[str, Any]] = []

    # 遍历其余视角作为 B
    for b in data[1:max_pairs_per_file + 1]:
        b_img_abs = Path(b["original"])
        b_mask_abs = Path(b["mask"])

        try:
            b_img_rel = b_img_abs.relative_to(root_dir)
        except ValueError:
            b_img_rel = b_img_abs

        mask = load_mask(b_mask_abs)
        if mask is None:
            continue

        pt = sample_point_from_mask(mask)
        if pt is None:
            continue

        x, y = pt
        fmt = "{:.3f}"
        ans = f"[({fmt.format(x)}, {fmt.format(y)})]"

        sid = str(b.get("scene_id", ref.get("scene_id", "")))
        uid = str(b.get("uid", ref.get("uid", "")))
        frame_a = str(ref.get("frame_id", ""))
        frame_b = str(b.get("frame_id", ""))

        sample_id = f"{sid}_uid{uid}_A{frame_a}_B{frame_b}"

        human = build_human_prompt(label)

        samples.append(
            {
                "id": sample_id,
                # 多图输入：A,B 两张图
                "image": [str(ref_img_rel), str(b_img_rel)],
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": ans},
                ],
            }
        )

    return samples


def collect_five_frames_files(
    root_dir: Path,
    split: str = "validation",
    max_files: int = 10,
) -> List[Path]:
    """
    在给定 split 下，收集若干 *_five_frames.json 文件。

    split:
      - "train" / "validation": 仅使用对应子目录
      - "both": 同时从 train 和 validation 收集（先train再val）
    """

    def collect_from(base: Path, remaining: int, acc: List[Path]) -> int:
        if not base.exists() or remaining <= 0:
            return remaining
        # 目录结构: base/scene_id/uid_xxx/*.json
        for scene_dir in sorted(base.iterdir()):
            if not scene_dir.is_dir():
                continue
            for uid_dir in sorted(scene_dir.iterdir()):
                if not uid_dir.is_dir():
                    continue
                for jf in uid_dir.glob("*_five_frames.json"):
                    acc.append(jf)
                    remaining -= 1
                    if remaining <= 0:
                        return 0
        return remaining

    files: List[Path] = []
    remaining = max_files

    if split in ("train", "validation"):
        base = root_dir / split
        collect_from(base, remaining, files)
    elif split == "both":
        # 先 train 再 validation
        remaining = collect_from(root_dir / "train", remaining, files)
        if remaining > 0:
            collect_from(root_dir / "validation", remaining, files)

    return files


def main():
    ap = argparse.ArgumentParser(
        description="从 five_frames 构造小规模 CrossView 多图 SFT JSON（测试用）"
    )
    ap.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="five_frames 数据根目录，例如 /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "both"],
        help="使用的划分（默认 validation, 也可 both 合并 train+validation）",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=500,
        help="最多使用多少个 *_five_frames.json 文件（每个文件最多生成若干对）",
    )
    ap.add_argument(
        "--max_pairs_per_file",
        type=int,
        default=4,
        help="每个 five_frames 文件最多生成多少 (A,B) 对（正例）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于mask采样）",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="输出 SFT JSON 路径（例如 tmp/crossview_fiveframes_sft_debug.json）",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root_dir = Path(args.root_dir)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_five_frames_files(root_dir, split=args.split, max_files=args.max_files)
    if not files:
        print(f"[WARN] 在 {root_dir}/{args.split} 下未找到 *_five_frames.json")
        return

    print(f"[INFO] 收集到 {len(files)} 个 five_frames 文件（用于小规模测试）")

    all_samples: List[Dict[str, Any]] = []
    for jf in files:
        s = build_sft_samples_from_five_frames(
            five_frames_path=jf,
            root_dir=root_dir,
            max_pairs_per_file=args.max_pairs_per_file,
        )
        print(f"[INFO] {jf}: 生成 {len(s)} 条样本")
        all_samples.extend(s)

    # 仅保存少量用于测试
    print(f"[INFO] 总共生成 {len(all_samples)} 条样本")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"[OK] 写入 {len(all_samples)} 条多图 SFT 样本 -> {out_path}")


if __name__ == "__main__":
    main()

