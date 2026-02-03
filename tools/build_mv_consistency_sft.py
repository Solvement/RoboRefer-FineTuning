#!/usr/bin/env python3
"""
构建多视角一致性SFT训练数据

从 five_frames 数据生成多图格式的训练数据：
- Image A: 标记了目标物体的参考图像（红色overlay）
- Image B: 查询图像（原始图像）
- GT: Image B中的归一化坐标或NOT_VISIBLE
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """读取mask为灰度图；读取失败返回None。"""
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return m


def compute_mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    计算mask的质心（归一化坐标）
    返回: (x_norm, y_norm) 或 None（如果mask为空）
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x_center = np.mean(xs)
    y_center = np.mean(ys)
    return (x_center / (w - 1), y_center / (h - 1))


def create_marked_image(original_path: Path, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    创建标记图像：在原始图像上叠加红色mask
    """
    img = Image.open(original_path).convert("RGB")
    img_array = np.array(img)
    
    # 创建红色overlay
    overlay = img_array.copy()
    overlay[mask > 0] = [255, 0, 0]  # 红色
    
    # 混合
    mask_3d = (mask > 0)[:, :, np.newaxis].astype(float)
    marked = (img_array * (1 - alpha * mask_3d) + overlay * (alpha * mask_3d)).astype(np.uint8)
    
    return Image.fromarray(marked)


def build_human_prompt() -> str:
    """构建human prompt（不包含<image> token，dataset会自动插入）"""
    return "You are given TWO separate images:\n- Image A (REFERENCE): the target object is highlighted (marked) in the image.\n- Image B (QUERY): you need to find the SAME object as in Image A.\n\nTASK:\n1. Look at Image A and understand which object is marked.\n2. Look at Image B and determine whether the SAME object is visible.\n3. If the object is visible in Image B, output ONE point coordinate on that object.\n4. If the object is NOT visible in Image B, answer NOT_VISIBLE.\n\nOUTPUT FORMAT:\n- If visible: answer with one coordinate in normalized [0,1] range relative to Image B only, in the form: [(x, y)]\n- If NOT visible: answer exactly: NOT_VISIBLE"


def find_frame_files(uid_dir: Path, k: int) -> Tuple[Optional[Path], Optional[Path]]:
    """
    查找第k个view的文件（k=1..5）
    返回: (original_path, mask_path) 或 (None, None)
    """
    k_str = f"{k:02d}"
    pattern_orig = f"{k_str}_*_original.png"
    pattern_mask_dilated = f"{k_str}_*_mask_dialated.png"
    pattern_mask = f"{k_str}_*_mask.png"
    
    orig_files = list(uid_dir.glob(pattern_orig))
    if not orig_files:
        return None, None
    
    orig_path = orig_files[0]
    
    # 优先使用mask_dialated，否则使用mask
    mask_dilated = list(uid_dir.glob(pattern_mask_dilated))
    mask_files = list(uid_dir.glob(pattern_mask))
    
    if mask_dilated:
        mask_path = mask_dilated[0]
    elif mask_files:
        mask_path = mask_files[0]
    else:
        mask_path = None
    
    return orig_path, mask_path


def extract_frame_id(filename: str) -> str:
    """从文件名提取frame_id（例如：01_004130_original.png -> 004130）"""
    match = re.search(r'_\d+_', filename)
    if match:
        return match.group(0)[1:-1]  # 去掉前后的下划线
    return "unknown"


def build_samples_for_uid(
    scene_id: str,
    uid: str,
    uid_dir: Path,
    marked_dir: Path,
    split: str,
    mode: str,
    anchor_k: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    为单个uid构建所有样本
    """
    random.seed(seed)
    
    # 收集5个view的数据
    views = {}
    for k in range(1, 6):
        orig_path, mask_path = find_frame_files(uid_dir, k)
        if orig_path is None:
            continue
        
        mask = None
        if mask_path:
            mask = load_mask(mask_path)
        
        if mask is None:
            continue
        
        # 计算GT点
        gt_point = compute_mask_centroid(mask)
        
        # 提取frame_id
        frame_id = extract_frame_id(orig_path.name)
        
        # 创建标记图像
        marked_rel = Path(split) / scene_id / f"uid_{uid}" / f"{k:02d}_{frame_id}_marked.png"
        marked_full = marked_dir / marked_rel
        marked_full.parent.mkdir(parents=True, exist_ok=True)
        
        marked_img = create_marked_image(orig_path, mask, alpha)
        marked_img.save(marked_full)
        
        views[k] = {
            "k": k,
            "frame_id": frame_id,
            "original_path": orig_path.resolve(),  # 绝对路径
            "mask": mask,
            "gt_point": gt_point,
            "marked_path": marked_full.resolve(),  # 绝对路径
        }
    
    if len(views) < 2:
        return []  # 至少需要2个view才能配对
    
    samples = []
    
    # 生成配对
    if mode == "anchor":
        # anchor模式：使用anchor_k作为A，其他view作为B
        if anchor_k not in views:
            return []
        
        ref_view = views[anchor_k]
        for k, query_view in views.items():
            if k == anchor_k:
                continue
            
            # 创建样本
            sample_id = f"{scene_id}_uid{uid}_A{ref_view['k']:02d}{ref_view['frame_id']}_B{query_view['k']:02d}{query_view['frame_id']}"
            
            gt_value = "NOT_VISIBLE" if query_view['gt_point'] is None else f"[({query_view['gt_point'][0]:.3f}, {query_view['gt_point'][1]:.3f})]"
            
            sample = {
                "id": sample_id,
                "image": [
                    str(ref_view['marked_path']),  # Image A: 标记图像（绝对路径）
                    str(query_view['original_path'])  # Image B: 原始图像（绝对路径）
                ],
                "conversations": [
                    {"from": "human", "value": build_human_prompt()},
                    {"from": "gpt", "value": gt_value}
                ]
            }
            samples.append(sample)
    
    elif mode == "allpairs":
        # allpairs模式：所有有向对
        for k_a, view_a in views.items():
            for k_b, view_b in views.items():
                if k_a == k_b:
                    continue
                
                sample_id = f"{scene_id}_uid{uid}_A{view_a['k']:02d}{view_a['frame_id']}_B{view_b['k']:02d}{view_b['frame_id']}"
                
                gt_value = "NOT_VISIBLE" if view_b['gt_point'] is None else f"[({view_b['gt_point'][0]:.3f}, {view_b['gt_point'][1]:.3f})]"
                
                sample = {
                    "id": sample_id,
                    "image": [
                        str(view_a['marked_path']),
                        str(view_b['original_path'])
                    ],
                    "conversations": [
                        {"from": "human", "value": build_human_prompt()},
                        {"from": "gpt", "value": gt_value}
                    ]
                }
                samples.append(sample)
    
    return samples


def add_hard_negatives(
    samples: List[Dict[str, Any]],
    scene_data: Dict[str, Dict[str, List[Dict]]],
    neg_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    添加hard negatives：使用相同的A，但B来自同一scene的不同uid
    """
    if neg_ratio <= 0:
        return samples
    
    random.seed(seed)
    negatives = []
    
    # 按scene_id分组samples
    samples_by_scene = {}
    for sample in samples:
        # 从id提取scene_id: {scene_id}_uid{uid}_...
        parts = sample['id'].split('_uid')
        if len(parts) < 2:
            continue
        scene_id = parts[0]
        if scene_id not in samples_by_scene:
            samples_by_scene[scene_id] = []
        samples_by_scene[scene_id].append(sample)
    
    for scene_id, scene_samples in samples_by_scene.items():
        if scene_id not in scene_data:
            continue
        
        # 收集该scene的所有uid的view信息
        other_uids = {}
        for uid, uid_views in scene_data[scene_id].items():
            if uid not in other_uids:
                other_uids[uid] = []
            for view in uid_views:
                other_uids[uid].append(view)
        
        for sample in scene_samples:
            # 提取当前sample的uid
            parts = sample['id'].split('_uid')
            if len(parts) < 2:
                continue
            current_uid = parts[1].split('_')[0]
            
            # 每个正样本添加floor(neg_ratio)个负样本
            n_neg = int(neg_ratio)
            if random.random() < (neg_ratio - n_neg):
                n_neg += 1
            
            for _ in range(n_neg):
                # 随机选择不同的uid
                available_uids = [uid for uid in other_uids.keys() if uid != current_uid and other_uids[uid]]
                if not available_uids:
                    break
                
                other_uid = random.choice(available_uids)
                other_views = other_uids[other_uid]
                if not other_views:
                    continue
                
                # 随机选择该uid的一个view作为B
                other_view = random.choice(other_views)
                other_orig = Path(other_view['original'])
                if not other_orig.exists():
                    continue
                
                # 创建负样本
                neg_id = sample['id'] + f"_NEG_{other_uid}"
                neg_sample = {
                    "id": neg_id,
                    "image": [
                        sample['image'][0],  # 相同的A
                        str(other_orig.resolve())  # 不同的B（绝对路径）
                    ],
                    "conversations": [
                        {"from": "human", "value": sample['conversations'][0]['value']},
                        {"from": "gpt", "value": "NOT_VISIBLE"}
                    ]
                }
                negatives.append(neg_sample)
    
    return samples + negatives


def main():
    parser = argparse.ArgumentParser(description="构建多视角一致性SFT训练数据")
    parser.add_argument("--data_root", type=str,
                       default="/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected",
                       help="数据根目录")
    parser.add_argument("--out_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--mode", type=str, default="anchor", choices=["anchor", "allpairs"],
                       help="配对模式：anchor或allpairs")
    parser.add_argument("--anchor_k", type=int, default=1,
                       help="anchor模式的参考view编号（1-5）")
    parser.add_argument("--neg_ratio", type=float, default=0.0,
                       help="负样本比例（hard negatives）")
    parser.add_argument("--alpha", type=float, default=0.45,
                       help="标记图像的overlay透明度")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    marked_dir = out_dir / "mv_marked_images_abs"
    marked_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 数据根目录: {data_root}")
    print(f"📁 输出目录: {out_dir}")
    print(f"🎯 模式: {args.mode}")
    if args.mode == "anchor":
        print(f"   Anchor view: {args.anchor_k}")
    print(f"📊 负样本比例: {args.neg_ratio}")
    print()
    
    all_samples_train = []
    all_samples_val = []
    
    # 处理train和validation
    for split in ["train", "validation"]:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"⚠️  Split目录不存在: {split_dir}")
            continue
        
        print(f"处理 {split} split...")
        
        # 收集scene数据（用于hard negatives）
        scene_data = {}
        
        samples = []
        scene_count = 0
        
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            scene_id = scene_dir.name
            scene_data[scene_id] = {}
            
            for uid_dir in sorted(scene_dir.iterdir()):
                if not uid_dir.is_dir() or not uid_dir.name.startswith("uid_"):
                    continue
                
                uid = uid_dir.name.replace("uid_", "")
                
                # 构建该uid的样本
                uid_samples = build_samples_for_uid(
                    scene_id, uid, uid_dir, marked_dir, split, args.mode, args.anchor_k, args.alpha, args.seed
                )
                
                # 收集view信息用于hard negatives
                views_list = []
                for k in range(1, 6):
                    orig_path, mask_path = find_frame_files(uid_dir, k)
                    if orig_path:
                        views_list.append({"original": str(orig_path)})
                scene_data[scene_id][uid] = views_list
                
                samples.extend(uid_samples)
            
            scene_count += 1
            if scene_count % 10 == 0:
                print(f"  已处理 {scene_count} 个场景...")
        
        # 添加hard negatives
        if args.neg_ratio > 0:
            print(f"  添加hard negatives...")
            samples = add_hard_negatives(samples, scene_data, args.neg_ratio, args.seed)
        
        if split == "train":
            all_samples_train = samples
        else:
            all_samples_val = samples
        
        print(f"✅ {split}: {len(samples)} 个样本")
    
    # 保存JSON
    train_json = out_dir / "mv_train.json"
    val_json = out_dir / "mv_val.json"
    
    with open(train_json, 'w') as f:
        json.dump(all_samples_train, f, indent=2, ensure_ascii=False)
    
    with open(val_json, 'w') as f:
        json.dump(all_samples_val, f, indent=2, ensure_ascii=False)
    
    print()
    print("="*70)
    print("✅ 数据生成完成！")
    print("="*70)
    print(f"训练数据: {train_json} ({len(all_samples_train)} 个样本)")
    print(f"验证数据: {val_json} ({len(all_samples_val)} 个样本)")
    print(f"标记图像目录: {marked_dir}")


if __name__ == "__main__":
    main()
