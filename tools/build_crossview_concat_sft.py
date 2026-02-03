#!/usr/bin/env python3
"""
从 five_frames 数据生成拼接图格式的 CrossView SFT 训练数据（只包含正例，不包含负例）

输入：
  - five_frames 数据目录
输出：
  - 拼接图图像目录
  - SFT训练JSON（拼接图格式，只包含正例）
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """读取mask为灰度图；读取失败返回None。"""
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return m


def sample_point_from_mask(mask: np.ndarray, use_center: bool = True) -> Optional[Tuple[float, float]]:
    """
    从mask中采样一个点（归一化坐标）
    如果use_center=True，使用distance transform找中心点；否则随机采样
    """
    if use_center:
        # 使用distance transform找中心点
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_loc = np.unravel_index(np.argmax(dist), dist.shape)
        y, x = max_loc
        h, w = mask.shape
        return (x / (w - 1), y / (h - 1))
    else:
        # 随机采样
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        i = np.random.randint(0, len(xs))
        y, x = ys[i], xs[i]
        h, w = mask.shape
        return (x / (w - 1), y / (h - 1))


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


def build_human_prompt(label: str) -> str:
    """构建human prompt"""
    return f"在Image A中用红色marker标记了目标物体。请在Image B中找到该物体，并输出其在Image B中的归一化坐标。如果该物体在Image B中不可见，请输出NOT_VISIBLE。"


def load_five_frames_data(root_dir: Path, split: str = "both") -> Dict[str, Dict[str, List[Dict]]]:
    """
    加载five_frames数据
    返回: {scene_id: {uid: [frame1, frame2, ...]}}
    """
    data = {}
    
    splits = ["train", "validation"] if split == "both" else [split]
    
    for split_name in splits:
        split_dir = root_dir / split_name
        if not split_dir.exists():
            print(f"⚠️  Split目录不存在: {split_dir}")
            continue
        
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            scene_id = scene_dir.name
            if scene_id not in data:
                data[scene_id] = {}
            
            for uid_dir in sorted(scene_dir.iterdir()):
                if not uid_dir.is_dir() or not uid_dir.name.startswith("uid_"):
                    continue
                
                uid = uid_dir.name.replace("uid_", "")
                json_file = uid_dir / f"{scene_id}_uid_{uid}_five_frames.json"
                
                if not json_file.exists():
                    continue
                
                with open(json_file, 'r') as f:
                    frames = json.load(f)
                
                if uid not in data[scene_id]:
                    data[scene_id][uid] = []
                
                # 确保路径是绝对路径
                for frame in frames:
                    if "original" in frame:
                        orig_path = Path(frame["original"])
                        if not orig_path.is_absolute():
                            frame["original"] = str(uid_dir / orig_path)
                        else:
                            frame["original"] = str(orig_path)
                    
                    if "mask" in frame:
                        mask_path = Path(frame["mask"])
                        if not mask_path.is_absolute():
                            frame["mask"] = str(uid_dir / mask_path)
                        else:
                            frame["mask"] = str(mask_path)
                
                data[scene_id][uid].extend(frames)
    
    return data


def build_concat_positives(
    data: Dict[str, Dict[str, List[Dict]]],
    root_dir: Path,
    concat_root: Path,
    max_pairs_per_uid: int = 8,
    min_mask_area: int = 100,
) -> List[Dict[str, Any]]:
    """
    构建拼接图格式的正例（只包含正例，不包含负例）
    """
    positives = []
    
    for scene_id, uids in data.items():
        for uid, frames in uids.items():
            if len(frames) < 2:
                continue
            
            # A: 第一个frame（标记了目标物体）
            ref = frames[0]
            ref_img = Path(ref["original"])
            label = ref.get("label", "")
            frame_a = str(ref.get("frame_id", ""))
            
            # B: 其余frames（最多max_pairs_per_uid个）
            available_b_frames = frames[1:]
            if len(available_b_frames) > max_pairs_per_uid:
                available_b_frames = random.sample(available_b_frames, max_pairs_per_uid)
            
            for b in available_b_frames:
                b_img = Path(b["original"])
                b_mask = Path(b["mask"])
                frame_b = str(b.get("frame_id", ""))
                
                # 检查B的mask
                mask = load_mask(b_mask)
                if mask is None:
                    continue
                
                # 过滤掉mask太小的正例
                mask_area = np.sum(mask > 0)
                if mask_area < min_mask_area:
                    continue
                
                # 使用distance transform找mask中心点（更确定性）
                pt = sample_point_from_mask(mask, use_center=True)
                if pt is None:
                    continue
                
                x, y = pt
                # 注意：拼接图中，B图在右侧，需要将B图的归一化坐标转换为拼接图的归一化坐标
                # 假设拼接图是A和B横向拼接，A在左，B在右
                # 如果A和B高度相同，那么B图的x坐标需要加上A图的宽度比例
                # 但这里我们直接输出B图的归一化坐标（在B图坐标系中）
                # 因为prompt中说的是"在Image B中的归一化坐标"
                ans = f"[({x:.3f}, {y:.3f})]"
                
                sample_id = f"{scene_id}_uid{uid}_A{frame_a}_B{frame_b}"
                
                # 生成拼接图
                # 构造相对路径（沿用 scene/uid/filename 结构）
                img_str = str(ref_img)
                if str(root_dir) in img_str:
                    rel = img_str.split(str(root_dir) + "/")[-1]
                else:
                    rel = ref_img.name
                
                rel_path = Path(rel)
                concat_rel = rel_path.with_name(rel_path.stem + f"_concat_A{frame_a}_B{frame_b}.png")
                concat_full = concat_root / concat_rel
                
                try:
                    make_concat_image(ref_img, b_img, concat_full)
                except Exception as e:
                    print(f"⚠️  拼接失败，跳过样本 {sample_id}: {e}")
                    continue
                
                human = build_human_prompt(label)
                
                # 拼接图格式：image字段是一个列表，但只包含一张拼接图的路径
                sample_dict = {
                    "id": sample_id,
                    "image": [str(concat_full)],  # 拼接图格式：只包含一张拼接图
                    "conversations": [
                        {"from": "human", "value": human},
                        {"from": "gpt", "value": ans}
                    ]
                }
                
                positives.append(sample_dict)
    
    return positives


def main():
    ap = argparse.ArgumentParser(description="生成拼接图格式的CrossView SFT训练数据（只包含正例）")
    ap.add_argument("--five_frames_root", required=True,
                   help="five_frames数据根目录，例如 /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected")
    ap.add_argument("--out_json", required=True,
                   help="输出SFT训练JSON路径")
    ap.add_argument("--concat_root", required=True,
                   help="拼接图输出根目录")
    ap.add_argument("--split", type=str, default="both", choices=["train", "validation", "both"],
                   help="使用哪个split的数据")
    ap.add_argument("--max_pairs_per_uid", type=int, default=8,
                   help="每个uid最多生成多少对(A,B)，默认8")
    ap.add_argument("--min_mask_area", type=int, default=100,
                   help="最小mask面积（像素），默认100")
    ap.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    
    args = ap.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    root_dir = Path(args.five_frames_root)
    if not root_dir.exists():
        print(f"❌ 数据根目录不存在: {root_dir}")
        return
    
    concat_root = Path(args.concat_root)
    concat_root.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 加载five_frames数据: {root_dir}")
    data = load_five_frames_data(root_dir, args.split)
    print(f"✅ 加载了 {len(data)} 个场景")
    
    print(f"🔨 构建拼接图正例（不包含负例）...")
    positives = build_concat_positives(
        data,
        root_dir,
        concat_root,
        max_pairs_per_uid=args.max_pairs_per_uid,
        min_mask_area=args.min_mask_area
    )
    
    print(f"✅ 生成了 {len(positives)} 个正例样本")
    
    # 保存JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(positives, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 数据保存完成！")
    print(f"   JSON文件: {out_json}")
    print(f"   拼接图目录: {concat_root}")
    print(f"   样本数: {len(positives)} (只包含正例)")


if __name__ == "__main__":
    main()
