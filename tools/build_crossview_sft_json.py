#!/usr/bin/env python3
"""
将CrossView question.json转换为RoboRefer SFT训练格式

输入：question.json（包含rgb_path, mask_path, gt_point_norm, gt_point_concat等）
输出：SFT训练JSON（包含id, image, conversations）
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import cv2


def load_mask(path: Path):
    """加载mask图像"""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return m


def in_mask(mask, x, y):
    """检查归一化坐标(x,y)是否在mask内"""
    h, w = mask.shape[:2]
    xx = int(round(x * (w - 1)))
    yy = int(round(y * (h - 1)))
    xx = max(0, min(w - 1, xx))
    yy = max(0, min(h - 1, yy))
    return mask[yy, xx] > 0


def sample_mask_point(mask):
    """从mask中随机采样一个点（返回full坐标）"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    i = np.random.randint(0, len(xs))
    y, x = ys[i], xs[i]
    h, w = mask.shape[:2]
    return (x / (w - 1), y / (h - 1))  # full coords in [0,1]


def full_to_right(x_full, y_full):
    """将full坐标（拼接图）转换为right坐标（View B）"""
    x_right = (x_full - 0.5) / 0.5
    return (max(0.0, min(1.0, x_right)), max(0.0, min(1.0, y_full)))


def main():
    ap = argparse.ArgumentParser(description="将CrossView question.json转换为RoboRefer SFT训练格式")
    ap.add_argument("--data_root", required=True, 
                   help="数据根目录，例如 Evaluation/RefSpatial-Bench/CrossView_corrected_x3_marker_v3_debug")
    ap.add_argument("--question_json", required=True, 
                   help="输入question.json路径，例如 question.json")
    ap.add_argument("--out_json", required=True,
                   help="输出SFT训练JSON路径")
    ap.add_argument("--coord_frame", choices=["right", "full"], default="right",
                   help="训练输出坐标系：right=相对右半张(ViewB)，full=相对拼接整图")
    ap.add_argument("--decimals", type=int, default=3,
                   help="坐标小数位数")
    ap.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    ap.add_argument("--neg_ratio", type=float, default=0.0,
                   help="负例比例(可选)，例如0.3表示30%负例")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    question_file = Path(args.question_json)
    
    # 加载question.json
    with open(question_file, 'r') as f:
        qs = json.load(f)

    print(f"加载了 {len(qs)} 个样本")

    # 用于构造负例：收集所有rgb_path
    rgb_list = [q["rgb_path"] for q in qs if "rgb_path" in q]

    out = []
    fmt = f"{{:.{args.decimals}f}}"

    for q in qs:
        img_rel = q["rgb_path"]
        mask_rel = q.get("mask_path", None)

        # 合并prompt和suffix（dataloader会自动处理<image> token）
        prompt = (q.get("prompt", "") or "").strip()
        suffix = (q.get("suffix", "") or "").strip()
        human = prompt + ("\n" + suffix if suffix else "")

        # 读取mask（拼接mask，只有右半有前景）
        mask = None
        if mask_rel:
            mask_path = data_root / mask_rel
            if mask_path.exists():
                mask = load_mask(mask_path)

        # 获取GT点（full和right两种都准备）
        gt_right = q.get("gt_point_norm", None)      # [x_b, y] - View B归一化坐标
        gt_full = q.get("gt_point_concat", None)    # [x_full, y] - 拼接图归一化坐标

        # 选一个"保证在mask内"的GT点（优先bbox center，否则从mask采样）
        chosen_full = None
        if mask is not None:
            # 先试bbox center（优先full，因为mask是full坐标）
            if gt_full is not None:
                if in_mask(mask, float(gt_full[0]), float(gt_full[1])):
                    chosen_full = (float(gt_full[0]), float(gt_full[1]))
            
            # bbox center不在mask内，从mask采样
            if chosen_full is None:
                s = sample_mask_point(mask)
                if s is not None:
                    chosen_full = (float(s[0]), float(s[1]))

        # 如果mask读不到/为空，退化用bbox center
        if chosen_full is None:
            if gt_full is not None:
                chosen_full = (float(gt_full[0]), float(gt_full[1]))
            elif gt_right is not None:
                # right -> full（需要知道img_a_width_norm，默认0.5）
                img_a_width_norm = q.get("img_a_width_norm", 0.5)
                x_full = img_a_width_norm + float(gt_right[0]) * (1 - img_a_width_norm)
                y_full = float(gt_right[1])
                chosen_full = (x_full, y_full)

        # 组装监督答案
        if chosen_full is None:
            ans = "NOT_VISIBLE"
        else:
            if args.coord_frame == "full":
                x, y = chosen_full
            else:
                # 转换为right坐标
                x, y = full_to_right(chosen_full[0], chosen_full[1])

            ans = f"[({fmt.format(x)}, {fmt.format(y)})]"

        out.append({
            "id": q.get("id", ""),
            "image": img_rel,   # 相对路径，dataloader会拼上image_path
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": ans}
            ]
        })

    # 负例：让模型学会NOT_VISIBLE
    if args.neg_ratio > 0 and len(rgb_list) > 1:
        n_neg = int(len(out) * args.neg_ratio)
        print(f"生成 {n_neg} 个负例样本")
        for i in range(n_neg):
            base = random.choice(qs)
            wrong = random.choice(rgb_list)
            while wrong == base["rgb_path"]:
                wrong = random.choice(rgb_list)

            prompt = (base.get("prompt", "") or "").strip()
            suffix = (base.get("suffix", "") or "").strip()
            human = prompt + ("\n" + suffix if suffix else "")

            out.append({
                "id": (base.get("id", "") or f"NEG{i}") + f"_NEG{i}",
                "image": wrong,
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": "NOT_VISIBLE"}
                ]
            })

    # 保存输出
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 写入 {len(out)} 个样本 -> {output_path}")
    print(f"  - 正例: {len(qs)}")
    if args.neg_ratio > 0:
        print(f"  - 负例: {len(out) - len(qs)}")
    print(f"  - 坐标系统: {args.coord_frame}")


if __name__ == "__main__":
    main()
