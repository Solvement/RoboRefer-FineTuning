#!/usr/bin/env python3
"""
生成多图CrossView SFT训练数据（包含高质量NOT_VISIBLE负例）

输入：
  - five_frames数据根目录
  - 或question.json（需要重构A/B路径）
  
输出：
  - crossview_multimg_sft_train.json（包含正例和3个tier的负例）
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import cv2
from collections import defaultdict

try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy未安装，将使用随机采样而非distance transform中心点")


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """加载mask图像"""
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return m


def sample_point_from_mask(mask: np.ndarray, use_center: bool = True) -> Optional[Tuple[float, float]]:
    """
    从mask中采样一个点（返回归一化坐标）
    
    Args:
        mask: 二值mask图像
        use_center: 如果True，使用distance transform找mask中心点（更确定性）
                    如果False，随机采样
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    
    h, w = mask.shape[:2]
    
    if use_center and HAS_SCIPY:
        # 使用distance transform找mask中心点（最远离边界的点）
        dist = distance_transform_edt(mask > 0)
        max_dist_idx = np.argmax(dist)
        y, x = np.unravel_index(max_dist_idx, mask.shape)
    else:
        # 随机采样
        i = np.random.randint(0, len(xs))
        y, x = ys[i], xs[i]
    
    return float(x) / float(w - 1), float(y) / float(h - 1)


def build_human_prompt(label: str) -> str:
    """
    构造多图cross-view指令（强约束版本，避免模型过度保守）
    """
    label_txt = label if label else "object"
    prompt = (
        "You are given TWO separate images:\n"
        "- Image A (REFERENCE): the target object is highlighted (marked) in the image.\n"
        "- Image B (QUERY): you need to find the SAME object as in Image A.\n\n"
        f"The target in Image A is a \"{label_txt}\". It is visually marked, so you can clearly see which object to track.\n\n"
        "TASK:\n"
        "You MUST choose exactly one of two outputs:\n"
        "(1) [(x, y)] if the same object is visible in Image B.\n"
        "(2) NOT_VISIBLE only if the object is definitely not visible in Image B.\n\n"
        "IMPORTANT:\n"
        "- If the object is visible in Image B, you MUST output a coordinate in the form: [(x, y)]\n"
        "- If you output NOT_VISIBLE while the object is visible, it is incorrect.\n"
        "- Only output NOT_VISIBLE if you are absolutely sure the object is not visible in Image B.\n\n"
        "OUTPUT FORMAT:\n"
        "- If visible: answer with one coordinate in normalized [0,1] range relative to Image B only, in the form: [(x, y)]\n"
        "- If NOT visible: answer exactly: NOT_VISIBLE\n"
    )
    return prompt


def load_five_frames_data(root_dir: Path, split: str = "both") -> Dict[str, List[Dict]]:
    """
    加载所有five_frames数据，按scene_id和uid组织
    
    返回: {scene_id: {uid: [frame_data, ...]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    
    splits = ["train", "validation"] if split == "both" else [split]
    
    for s in splits:
        split_dir = root_dir / s
        if not split_dir.exists():
            continue
            
        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            
            for uid_dir in scene_dir.iterdir():
                if not uid_dir.is_dir() or not uid_dir.name.startswith("uid_"):
                    continue
                
                uid = uid_dir.name.replace("uid_", "")
                json_file = uid_dir / f"{scene_id}_uid_{uid}_five_frames.json"
                
                if json_file.exists():
                    try:
                        frames = json.loads(json_file.read_text())
                        if isinstance(frames, list):
                            data[scene_id][uid].extend(frames)
                    except Exception as e:
                        print(f"⚠️  读取失败 {json_file}: {e}")
    
    return data


def build_visibility_index(data: Dict[str, Dict[str, List[Dict]]], root_dir: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    构建可见性索引：{scene_id: {image_path: set(visible_uids)}}
    
    修复：基于图像路径而不是frame_id，因为：
    - 同一个frame_id可能对应多个不同的图像路径（不同UID）
    - 需要检查实际图像路径来确定哪些UID在这个图像中可见
    """
    index = defaultdict(lambda: defaultdict(set))
    
    for scene_id, uids in data.items():
        for uid, frames in uids.items():
            for frame in frames:
                # 使用图像路径作为key，而不是frame_id
                img_path = frame.get("original", "")
                if not img_path:
                    continue
                
                # 转换为相对路径（相对于root_dir）或使用绝对路径
                try:
                    img_path_obj = Path(img_path)
                    if img_path_obj.is_absolute():
                        # 尝试转换为相对路径
                        try:
                            rel_path = str(img_path_obj.relative_to(root_dir))
                        except ValueError:
                            # 如果不在root_dir下，使用绝对路径
                            rel_path = str(img_path_obj)
                    else:
                        rel_path = str(img_path_obj)
                except:
                    rel_path = str(img_path)
                
                mask_path = root_dir / frame.get("mask", "")
                
                if mask_path.exists():
                    mask = load_mask(mask_path)
                    if mask is not None and np.any(mask > 0):
                        # 使用图像路径作为key
                        index[scene_id][rel_path].add(uid)
    
    return index


def get_depth_path(image_path: Path, depth_root: Optional[Path] = None, depth_map: Optional[dict] = None) -> Optional[str]:
    """
    根据RGB图像路径构造depth路径
    
    Args:
        image_path: RGB图像路径（可以是绝对路径或相对路径）
        depth_root: depth数据根目录
        depth_map: depth映射字典 {rgb_rel_path: depth_rel_path}
    
    Returns:
        depth图像路径（相对路径），如果不存在返回None
    """
    # 优先使用depth_map
    if depth_map is not None:
        # 尝试匹配image_path
        image_path_str = str(image_path)
        # 计算相对路径用于匹配
        try:
            parts = image_path.parts
            if 'train' in parts or 'validation' in parts:
                split_idx = next(i for i, p in enumerate(parts) if p in ['train', 'validation'])
                rgb_rel = str(Path(*parts[split_idx:]))
            else:
                rgb_rel = image_path.name
            
            # 尝试多种匹配方式
            for map_rgb, map_depth in depth_map.items():
                # 直接匹配
                if rgb_rel == map_rgb or image_path_str.endswith(map_rgb):
                    if depth_root:
                        depth_path = depth_root / map_depth
                    else:
                        depth_path = Path(map_depth)
                    if depth_path.exists():
                        return str(map_depth) if depth_root else str(depth_path)
                
                # 尝试匹配文件名（处理_original.png vs _inpainted.png的情况）
                if rgb_rel.endswith('_original.png'):
                    # 尝试将_original.png替换为其他后缀来匹配
                    base_name = rgb_rel.replace('_original.png', '')
                    if map_rgb.startswith(base_name) and map_rgb.endswith('.png'):
                        # 使用对应的depth
                        if depth_root:
                            depth_path = depth_root / map_depth
                        else:
                            depth_path = Path(map_depth)
                        if depth_path.exists():
                            return str(map_depth) if depth_root else str(depth_path)
        except:
            pass
    
    # 如果没有depth_map，尝试从image_path推断
    if depth_root is None:
        # 尝试从image_path推断depth路径
        # 例如: .../01_001640_original.png -> .../01_001640_depth.png
        depth_path = image_path.parent / image_path.name.replace("_original.png", "_depth.png")
    else:
        # 使用depth_root构造路径
        # 尝试计算相对路径
        try:
            # 假设image_path是绝对路径，尝试找到相对于某个根目录的路径
            # 对于five_frames数据，结构是: root/split/scene/uid/file
            parts = image_path.parts
            # 找到包含split的部分
            if 'train' in parts or 'validation' in parts:
                split_idx = next(i for i, p in enumerate(parts) if p in ['train', 'validation'])
                rel_path = Path(*parts[split_idx:])
            else:
                # 使用最后3层目录结构
                rel_path = Path(*parts[-3:])
            
            # 构造depth路径：将_original.png替换为_depth.png
            depth_rel = rel_path.parent / rel_path.name.replace("_original.png", "_depth.png")
            depth_path = depth_root / depth_rel
        except:
            # 如果推断失败，尝试直接替换文件名
            depth_path = depth_root / image_path.name.replace("_original.png", "_depth.png")
    
    # 检查文件是否存在
    if depth_path.exists():
        # 返回相对路径（相对于depth_root）
        if depth_root:
            try:
                return str(depth_path.relative_to(depth_root))
            except:
                return str(depth_path)
        return str(depth_path)
    return None

def build_positives(
    data: Dict[str, Dict[str, List[Dict]]],
    root_dir: Path,
    max_pairs_per_uid: int = 4,
    min_mask_area: int = 100,
    filter_top_percentile: Optional[float] = None,
    depth_root: Optional[Path] = None,
    depth_map: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    构建正例：A和B都包含同一uid
    """
    positives = []
    
    for scene_id, uids in data.items():
        for uid, frames in uids.items():
            if len(frames) < 2:
                continue
            
            # 改进：每个frame都可以作为A，增加数据多样性
            # 但为了保持一致性，仍然优先使用第一个frame作为A
            # 如果frames数量多，可以生成更多对
            
            # A: 第一个frame（主要）
            ref = frames[0]
            ref_img = Path(ref["original"])
            label = ref.get("label", "")
            frame_a = str(ref.get("frame_id", ""))
            
            # 使用绝对路径，因为dataloader需要完整路径
            ref_img_rel = str(ref_img)
            
            # B: 其余frames（最多max_pairs_per_uid个）
            available_b_frames = frames[1:]
            if len(available_b_frames) > max_pairs_per_uid:
                # 如果B frames太多，随机选择max_pairs_per_uid个
                available_b_frames = random.sample(available_b_frames, max_pairs_per_uid)
            
            for b in available_b_frames:
                b_img = Path(b["original"])
                b_mask = Path(b["mask"])
                frame_b = str(b.get("frame_id", ""))
                
                # 使用绝对路径
                b_img_rel = str(b_img)
                
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
                ans = f"[({x:.3f}, {y:.3f})]"
                
                sample_id = f"{scene_id}_uid{uid}_A{frame_a}_B{frame_b}"
                human = build_human_prompt(label)
                
                # 构造depth路径（如果启用depth）
                depth_list = None
                if depth_root is not None:
                    ref_depth = get_depth_path(Path(ref_img_rel), depth_root, depth_map)
                    b_depth = get_depth_path(Path(b_img_rel), depth_root, depth_map)
                    if ref_depth and b_depth:
                        depth_list = [ref_depth, b_depth]
                    elif ref_depth or b_depth:
                        # 如果只有一个depth，仍然使用（另一个可能不存在）
                        depth_list = [ref_depth or "", b_depth or ""]
                
                # 存储mask面积用于后续过滤
                sample_dict = {
                    "id": sample_id,
                    "image": [str(ref_img_rel), str(b_img_rel)],
                    "conversations": [
                        {"from": "human", "value": human},
                        {"from": "gpt", "value": ans}
                    ],
                    "meta": {
                        "scene": scene_id,
                        "frameA": frame_a,
                        "frameB": frame_b,
                        "uid": uid,
                        "is_neg": False,
                        "neg_type": None,
                        "mask_area": int(mask_area)  # 存储mask面积
                    }
                }
                
                # 如果启用depth，添加depth字段
                if depth_list:
                    sample_dict["depth"] = depth_list
                
                positives.append(sample_dict)
    
    # 如果指定了filter_top_percentile，只保留mask面积最大的样本
    if filter_top_percentile is not None and len(positives) > 0:
        # 按mask面积排序
        positives.sort(key=lambda x: x["meta"]["mask_area"], reverse=True)
        # 保留top percentile
        n_keep = int(len(positives) * filter_top_percentile)
        positives = positives[:n_keep]
        print(f"   ✅ 过滤后保留 {len(positives)} 个正例（top {filter_top_percentile*100:.0f}%）")
    
    return positives


def build_tier_a_negatives(
    positives: List[Dict[str, Any]],
    data: Dict[str, Dict[str, List[Dict]]],
    root_dir: Path,
    n_neg: int,
    depth_root: Optional[Path] = None,
    depth_map: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Tier A: Easy negatives (cross-scene mismatches)
    取一个正例的A，但B来自不同scene
    """
    negatives = []
    scenes = list(data.keys())
    
    if len(scenes) < 2:
        print("⚠️  场景数不足，无法生成Tier A负例")
        return []
    
    for _ in range(n_neg):
        # 随机选一个正例
        pos = random.choice(positives)
        scene_a = pos["meta"]["scene"]
        uid = pos["meta"]["uid"]
        frame_a = pos["meta"]["frameA"]
        
        # 选一个不同scene的B
        other_scenes = [s for s in scenes if s != scene_a]
        if not other_scenes:
            continue
        
        scene_b = random.choice(other_scenes)
        
        # 从scene_b随机选一个frame作为B
        if scene_b not in data or not data[scene_b]:
            continue
        
        # 随机选一个uid和frame
        random_uid = random.choice(list(data[scene_b].keys()))
        random_frames = data[scene_b][random_uid]
        if not random_frames:
            continue
        
        b_frame = random.choice(random_frames)
        b_img = Path(b_frame["original"])
        frame_b = str(b_frame.get("frame_id", ""))
        
        # 使用绝对路径
        b_img_rel = str(b_img)
        
        # 保持A不变
        ref_img_rel = pos["image"][0]
        label = pos["conversations"][0]["value"].split('"')[1] if '"' in pos["conversations"][0]["value"] else "object"
        
        sample_id = f"{scene_a}_uid{uid}_A{frame_a}_B{scene_b}_{frame_b}_TIERA"
        human = build_human_prompt(label)
        
        # 构造depth路径（如果启用depth）
        depth_list = None
        if depth_root is not None or depth_map is not None:
            ref_depth = get_depth_path(Path(ref_img_rel), depth_root, depth_map)
            b_depth = get_depth_path(Path(b_img_rel), depth_root, depth_map)
            if ref_depth and b_depth:
                depth_list = [ref_depth, b_depth]
        
        neg_dict = {
            "id": sample_id,
            "image": [ref_img_rel, str(b_img_rel)],
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": "NOT_VISIBLE"}
            ],
            "meta": {
                "scene": scene_a,
                "frameA": frame_a,
                "frameB": frame_b,
                "uid": uid,
                "is_neg": True,
                "neg_type": "tierA"
            }
        }
        
        if depth_list:
            neg_dict["depth"] = depth_list
        
        negatives.append(neg_dict)
    
    return negatives


def build_tier_b_negatives(
    positives: List[Dict[str, Any]],
    data: Dict[str, Dict[str, List[Dict]]],
    root_dir: Path,
    visibility_index: Dict[str, Dict[str, Set[str]]],
    n_neg: int,
    depth_root: Optional[Path] = None,
    depth_map: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Tier B: Medium negatives (same scene, wrong uid in B)
    A标记uid1，但B中不存在uid1
    
    修复：确保B帧的图像路径中不包含uid_a
    """
    negatives = []
    
    for _ in range(n_neg):
        # 随机选一个正例
        pos = random.choice(positives)
        scene = pos["meta"]["scene"]
        uid_a = pos["meta"]["uid"]
        frame_a = pos["meta"]["frameA"]
        
        if scene not in data:
            continue
        
        scene_uids = set(data[scene].keys())
        # 排除uid_a
        candidate_uids = scene_uids - {uid_a}
        if not candidate_uids:
            continue
        
        # 尝试多次，找到一个B帧确实不包含uid_a的
        max_attempts = 20
        found_valid = False
        
        for attempt in range(max_attempts):
            # 随机选一个不同的uid
            wrong_uid = random.choice(list(candidate_uids))
            
            if wrong_uid not in data[scene]:
                continue
            
            wrong_frames = data[scene][wrong_uid]
            if not wrong_frames:
                continue
            
            b_frame = random.choice(wrong_frames)
            b_img_path = b_frame["original"]
            b_img = Path(b_img_path)
            frame_b_new = str(b_frame.get("frame_id", ""))
            
            # 检查B帧的图像路径中是否包含uid_a
            # 方法1: 检查visibility_index（基于图像路径）
            try:
                b_img_rel = str(b_img.relative_to(root_dir))
            except ValueError:
                b_img_rel = str(b_img)
            
            # 检查这个B帧图像中可见的UID
            visible_uids_in_b = visibility_index.get(scene, {}).get(b_img_rel, set())
            
            # 如果B帧中uid_a可见，跳过这个候选
            if uid_a in visible_uids_in_b:
                continue
            
            # 方法2: 检查B帧的图像路径是否在uid_a的frames中
            # 如果B帧的图像路径和uid_a的某个frame相同，说明B帧包含uid_a
            uid_a_frames = data[scene].get(uid_a, [])
            b_img_abs = str(b_img) if b_img.is_absolute() else str(root_dir / b_img)
            uid_a_has_same_image = any(
                str(Path(f.get("original", ""))) == b_img_abs 
                for f in uid_a_frames
            )
            
            if uid_a_has_same_image:
                continue
            
            # 找到了一个有效的负例
            found_valid = True
            
            # 保持A不变（标记的是uid_a）
            ref_img_rel = pos["image"][0]
            label = pos["conversations"][0]["value"].split('"')[1] if '"' in pos["conversations"][0]["value"] else "object"
            
            sample_id = f"{scene}_uid{uid_a}_A{frame_a}_B{frame_b_new}_TIERB"
            human = build_human_prompt(label)
            
            # 构造depth路径（如果启用depth）
            depth_list = None
            if depth_root is not None or depth_map is not None:
                ref_depth = get_depth_path(Path(ref_img_rel), depth_root, depth_map)
                b_depth = get_depth_path(b_img, depth_root, depth_map)
                if ref_depth and b_depth:
                    depth_list = [ref_depth, b_depth]
            
            neg_dict = {
                "id": sample_id,
                "image": [ref_img_rel, str(b_img)],
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": "NOT_VISIBLE"}
                ],
                "meta": {
                    "scene": scene,
                    "frameA": frame_a,
                    "frameB": frame_b_new,
                    "uid": uid_a,
                    "is_neg": True,
                    "neg_type": "tierB"
                }
            }
            
            if depth_list:
                neg_dict["depth"] = depth_list
            
            negatives.append(neg_dict)
            break
        
        # 如果尝试多次都没找到有效的，跳过这个正例
        if not found_valid:
            continue
    
    return negatives


def build_tier_c_negatives(
    positives: List[Dict[str, Any]],
    data: Dict[str, Dict[str, List[Dict]]],
    root_dir: Path,
    visibility_index: Dict[str, Dict[str, Set[str]]],
    n_neg: int,
    depth_root: Optional[Path] = None,
    depth_map: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Tier C: Hard negatives (same scene, same uid, but not visible in B)
    A包含uid，但B的mask中该uid不可见（occluded/out-of-view/filtered）
    
    修复：基于图像路径检查可见性
    """
    negatives = []
    
    # 先找出所有可能的hard negative候选
    candidates = []
    
    for scene_id, uids in data.items():
        for uid, frames in uids.items():
            if len(frames) < 2:
                continue
            
            # A: 第一个frame（包含uid）
            ref = frames[0]
            frame_a = str(ref.get("frame_id", ""))
            ref_img_path = ref.get("original", "")
            
            # 转换为相对路径用于visibility_index查找
            try:
                ref_img_rel = str(Path(ref_img_path).relative_to(root_dir))
            except ValueError:
                ref_img_rel = str(ref_img_path)
            
            # 检查其他frames，看哪些不包含该uid
            for b in frames[1:]:
                frame_b = str(b.get("frame_id", ""))
                b_img_path = b.get("original", "")
                
                # 转换为相对路径用于visibility_index查找
                try:
                    b_img_rel = str(Path(b_img_path).relative_to(root_dir))
                except ValueError:
                    b_img_rel = str(b_img_path)
                
                # 使用图像路径检查可见性
                visible_uids = visibility_index.get(scene_id, {}).get(b_img_rel, set())
                
                if uid not in visible_uids:
                    # 这是一个hard negative候选
                    candidates.append({
                        "scene": scene_id,
                        "uid": uid,
                        "frameA": frame_a,
                        "frameB": frame_b,
                        "ref": ref,
                        "b": b
                    })
    
    if len(candidates) < n_neg:
        print(f"⚠️  Hard negative候选数({len(candidates)})少于需求({n_neg})")
        n_neg = len(candidates)
    
    selected = random.sample(candidates, min(n_neg, len(candidates)))
    
    for cand in selected:
        ref = cand["ref"]
        b = cand["b"]
        
        ref_img = Path(ref["original"])
        b_img = Path(b["original"])
        label = ref.get("label", "")
        frame_a = cand["frameA"]
        frame_b = cand["frameB"]
        
        # 使用绝对路径
        ref_img_rel = str(ref_img)
        b_img_rel = str(b_img)
        
        sample_id = f"{cand['scene']}_uid{cand['uid']}_A{frame_a}_B{frame_b}_TIERC"
        human = build_human_prompt(label)
        
        # 构造depth路径（如果启用depth）
        depth_list = None
        if depth_root is not None or depth_map is not None:
            ref_depth = get_depth_path(Path(ref_img_rel), depth_root, depth_map)
            b_depth = get_depth_path(Path(b_img_rel), depth_root, depth_map)
            if ref_depth and b_depth:
                depth_list = [ref_depth, b_depth]
        
        neg_dict = {
            "id": sample_id,
            "image": [str(ref_img_rel), str(b_img_rel)],
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": "NOT_VISIBLE"}
            ],
            "meta": {
                "scene": cand["scene"],
                "frameA": frame_a,
                "frameB": frame_b,
                "uid": cand["uid"],
                "is_neg": True,
                "neg_type": "tierC"
            }
        }
        
        if depth_list:
            neg_dict["depth"] = depth_list
        
        negatives.append(neg_dict)
    
    return negatives


def main():
    ap = argparse.ArgumentParser(description="生成多图CrossView SFT训练数据（包含NOT_VISIBLE负例）")
    ap.add_argument("--five_frames_root", required=True,
                   help="five_frames数据根目录，例如 /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3")
    ap.add_argument("--out_json", required=True,
                   help="输出SFT训练JSON路径")
    ap.add_argument("--neg_ratio", type=float, default=0.15,
                   help="负例比例，例如0.15表示15%%负例（降低easy negative，提升hard negative）")
    ap.add_argument("--neg_tiers", type=str, default="40,40,20",
                   help="负例tier分布（Tier A, B, C的百分比），例如'40,40,20'（降低Tier C，提升A/B）")
    ap.add_argument("--filter_top_percentile", type=float, default=None,
                   help="只使用mask面积top X%的正例（用于curriculum learning），例如0.5表示top 50%")
    ap.add_argument("--curriculum_phase", type=str, default=None, choices=["phase1", "phase2"],
                   help="Curriculum learning阶段：phase1=0%%负例，phase2=正常负例比例")
    ap.add_argument("--depth_root", type=str, default=None,
                   help="Depth数据根目录（如果提供，会在JSON中添加depth字段）")
    ap.add_argument("--split", type=str, default="both", choices=["train", "validation", "both"],
                   help="使用哪个split的数据")
    ap.add_argument("--max_pairs_per_uid", type=int, default=8,
                   help="每个uid最多生成多少对(A,B)，默认8（增加训练数据量）")
    ap.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    
    args = ap.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    root_dir = Path(args.five_frames_root)
    if not root_dir.exists():
        print(f"❌ 数据根目录不存在: {root_dir}")
        return
    
    print(f"📂 加载five_frames数据: {root_dir}")
    data = load_five_frames_data(root_dir, args.split)
    print(f"✅ 加载了 {len(data)} 个场景")
    
    print(f"🔍 构建可见性索引...")
    visibility_index = build_visibility_index(data, root_dir)
    print(f"✅ 索引构建完成")
    
    # 处理depth_root和depth_map
    depth_root = Path(args.depth_root) if args.depth_root else None
    depth_map = None
    
    if depth_root:
        print(f"📊 Depth数据根目录: {depth_root}")
        if not depth_root.exists():
            print(f"⚠️  Depth目录不存在，将跳过depth字段")
            depth_root = None
        else:
            # 尝试加载depth_map.json
            depth_map_file = depth_root / "depth_map.json"
            if depth_map_file.exists():
                with open(depth_map_file) as f:
                    depth_map = json.load(f)
                print(f"📊 加载depth映射文件: {len(depth_map)} 条映射")
    
    print(f"📊 构建正例...")
    # Curriculum learning: Phase 1使用top 50%大目标，Phase 2使用全部
    filter_percentile = args.filter_top_percentile
    if args.curriculum_phase == "phase1":
        if filter_percentile is None:
            filter_percentile = 0.5  # 默认使用top 50%
        print(f"   📚 Curriculum Phase 1: 只使用mask面积top {filter_percentile*100:.0f}%的正例")
    elif args.curriculum_phase == "phase2":
        filter_percentile = None  # Phase 2使用全部正例
        print(f"   📚 Curriculum Phase 2: 使用全部正例")
    
    positives = build_positives(data, root_dir, args.max_pairs_per_uid, 
                                filter_top_percentile=filter_percentile,
                                depth_root=depth_root,
                                depth_map=depth_map)
    print(f"✅ 生成了 {len(positives)} 个正例")
    
    # Curriculum learning: Phase 1不使用负例
    if args.curriculum_phase == "phase1":
        print(f"\n📚 Curriculum Phase 1: 跳过负例生成（neg_ratio=0）")
        n_neg_total = 0
        args.neg_ratio = 0.0
    else:
        # 计算负例数量
        n_neg_total = int(len(positives) * args.neg_ratio)
    tier_percents = [float(x) for x in args.neg_tiers.split(",")]
    tier_percents = [p / sum(tier_percents) for p in tier_percents]  # 归一化
    
    n_tier_a = int(n_neg_total * tier_percents[0])
    n_tier_b = int(n_neg_total * tier_percents[1])
    n_tier_c = n_neg_total - n_tier_a - n_tier_b
    
    print(f"\n📊 构建负例 (总计 {n_neg_total} 个):")
    print(f"   Tier A (easy): {n_tier_a}")
    print(f"   Tier B (medium): {n_tier_b}")
    print(f"   Tier C (hard): {n_tier_c}")
    
    negatives = []
    
    if n_neg_total == 0:
        print(f"   ⏭️  跳过负例生成（Curriculum Phase 1）")
    elif n_tier_a > 0:
        print(f"   生成Tier A负例...")
        tier_a = build_tier_a_negatives(positives, data, root_dir, n_tier_a, depth_root=depth_root, depth_map=depth_map)
        negatives.extend(tier_a)
        print(f"   ✅ 生成了 {len(tier_a)} 个Tier A负例")
    
    if n_tier_b > 0:
        print(f"   生成Tier B负例...")
        tier_b = build_tier_b_negatives(positives, data, root_dir, visibility_index, n_tier_b, depth_root=depth_root, depth_map=depth_map)
        negatives.extend(tier_b)
        print(f"   ✅ 生成了 {len(tier_b)} 个Tier B负例")
    
    if n_tier_c > 0:
        print(f"   生成Tier C负例...")
        tier_c = build_tier_c_negatives(positives, data, root_dir, visibility_index, n_tier_c, depth_root=depth_root, depth_map=depth_map)
        negatives.extend(tier_c)
        print(f"   ✅ 生成了 {len(tier_c)} 个Tier C负例")
    
    # 合并
    all_samples = positives + negatives
    random.shuffle(all_samples)
    
    # 保存
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成！")
    print(f"   - 总样本数: {len(all_samples)}")
    print(f"   - 正例: {len(positives)} ({len(positives)/len(all_samples)*100:.1f}%%)")
    print(f"   - 负例: {len(negatives)} ({len(negatives)/len(all_samples)*100:.1f}%%)")
    print(f"   - Tier A: {len([n for n in negatives if n['meta']['neg_type'] == 'tierA'])}")
    print(f"   - Tier B: {len([n for n in negatives if n['meta']['neg_type'] == 'tierB'])}")
    print(f"   - Tier C: {len([n for n in negatives if n['meta']['neg_type'] == 'tierC'])}")
    print(f"   - 输出: {output_path}")


if __name__ == "__main__":
    main()
