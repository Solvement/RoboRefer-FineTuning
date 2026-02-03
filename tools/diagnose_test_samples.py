#!/usr/bin/env python3
"""
诊断测试样本：检查B是否真的可见，路径映射是否正确
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

def load_mask(mask_path: Path) -> np.ndarray:
    """加载mask图像"""
    if not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return m

def build_visibility_index(fiveframes_root: Path) -> dict:
    """
    构建可见性索引：{scene_id: {frame_id: set(visible_uids)}}
    """
    index = defaultdict(lambda: defaultdict(set))
    
    for split in ["train", "validation"]:
        split_dir = fiveframes_root / split
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
                            for frame in frames:
                                frame_id = str(frame.get("frame_id", ""))
                                mask_path = fiveframes_root / frame.get("mask", "")
                                if mask_path.exists():
                                    mask = load_mask(mask_path)
                                    if mask is not None and np.any(mask > 0):
                                        index[scene_id][frame_id].add(uid)
                    except Exception as e:
                        print(f"⚠️  读取失败 {json_file}: {e}", file=sys.stderr)
    
    return index

def find_fiveframes_image(
    fiveframes_root: Path,
    scene_id: str,
    uid: str,
    frame_id: str
) -> tuple:
    """
    在 five_frames 数据中查找对应的图像和mask
    
    Returns:
        (image_path, mask_path) 或 (None, None)
    """
    for split in ["train", "validation"]:
        uid_dir = fiveframes_root / split / scene_id / f"uid_{uid}"
        if not uid_dir.exists():
            continue
        
        json_file = uid_dir / f"{scene_id}_uid_{uid}_five_frames.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    frames_data = json.load(f)
                for frame_data in frames_data:
                    if str(frame_data.get("frame_id", "")) == str(frame_id):
                        original_path = Path(frame_data.get("original", ""))
                        mask_path = Path(frame_data.get("mask", ""))
                        
                        # 尝试绝对路径
                        if original_path.exists():
                            return (original_path, mask_path if mask_path.exists() else None)
                        # 尝试相对路径
                        rel_img = uid_dir / original_path.name
                        rel_mask = uid_dir / mask_path.name if mask_path else None
                        if rel_img.exists():
                            return (rel_img, rel_mask if rel_mask and rel_mask.exists() else None)
            except Exception as e:
                print(f"⚠️  读取 {json_file} 失败: {e}", file=sys.stderr)
                continue
    
    return (None, None)

def diagnose_samples(question_json: Path, fiveframes_root: Path, max_samples: int = 10):
    """
    诊断测试样本
    """
    # 加载测试数据
    with open(question_json) as f:
        questions = json.load(f)
    
    # 限制样本数
    questions = questions[:max_samples]
    
    # 构建可见性索引
    print("🔍 构建可见性索引...")
    visibility_index = build_visibility_index(fiveframes_root)
    print(f"✅ 索引构建完成")
    
    # 诊断每个样本
    print(f"\n📊 诊断 {len(questions)} 个测试样本：\n")
    print("id | A_path_exists | B_path_exists | B_mask_nonzero | uid_in_B_visible_set | B_mask_area")
    print("-" * 100)
    
    results = []
    
    for q in questions:
        sample_id = q.get("id", "N/A")
        scene_id = q.get("scene_id", "")
        uid = q.get("uid", "")
        frame_a_id = q.get("frame_a_id", "")
        frame_b_id = q.get("frame_b_id", "")
        
        # 查找A和B的图像路径
        a_img_path, a_mask_path = find_fiveframes_image(fiveframes_root, scene_id, uid, frame_a_id)
        b_img_path, b_mask_path = find_fiveframes_image(fiveframes_root, scene_id, uid, frame_b_id)
        
        # 检查路径存在性
        a_path_exists = a_img_path is not None and a_img_path.exists()
        b_path_exists = b_img_path is not None and b_img_path.exists()
        
        # 检查B的mask
        b_mask_nonzero = 0
        b_mask_area = 0
        if b_mask_path and b_mask_path.exists():
            mask = load_mask(b_mask_path)
            if mask is not None:
                b_mask_nonzero = np.sum(mask > 0)
                b_mask_area = b_mask_nonzero
        
        # 检查uid是否在B的visible set中
        uid_in_b_visible = False
        if scene_id in visibility_index and frame_b_id in visibility_index[scene_id]:
            uid_in_b_visible = uid in visibility_index[scene_id][frame_b_id]
        
        # 输出结果
        print(f"{sample_id[:50]} | {a_path_exists} | {b_path_exists} | {b_mask_nonzero} | {uid_in_b_visible} | {b_mask_area}")
        
        results.append({
            "id": sample_id,
            "scene_id": scene_id,
            "uid": uid,
            "frame_a_id": frame_a_id,
            "frame_b_id": frame_b_id,
            "a_path": str(a_img_path) if a_img_path else None,
            "b_path": str(b_img_path) if b_img_path else None,
            "b_mask_path": str(b_mask_path) if b_mask_path else None,
            "a_path_exists": a_path_exists,
            "b_path_exists": b_path_exists,
            "b_mask_nonzero": int(b_mask_nonzero),
            "b_mask_area": int(b_mask_area),
            "uid_in_b_visible_set": uid_in_b_visible
        })
    
    # 统计
    print("\n" + "=" * 100)
    print("📈 统计结果：")
    print(f"  总样本数: {len(results)}")
    print(f"  A路径存在: {sum(1 for r in results if r['a_path_exists'])} ({sum(1 for r in results if r['a_path_exists'])/len(results)*100:.1f}%)")
    print(f"  B路径存在: {sum(1 for r in results if r['b_path_exists'])} ({sum(1 for r in results if r['b_path_exists'])/len(results)*100:.1f}%)")
    print(f"  B_mask非空: {sum(1 for r in results if r['b_mask_nonzero'] > 0)} ({sum(1 for r in results if r['b_mask_nonzero'] > 0)/len(results)*100:.1f}%)")
    print(f"  uid在B可见: {sum(1 for r in results if r['uid_in_b_visible_set'])} ({sum(1 for r in results if r['uid_in_b_visible_set'])/len(results)*100:.1f}%)")
    
    # 关键诊断
    print("\n🔍 关键诊断：")
    invisible_count = sum(1 for r in results if r['b_mask_nonzero'] == 0 or not r['uid_in_b_visible_set'])
    if invisible_count > len(results) * 0.5:
        print(f"  ⚠️  警告：{invisible_count}/{len(results)} 个样本的B不可见或uid不在visible set中")
        print(f"  → 这可能是导致模型输出NOT_VISIBLE的原因（模型行为可能是正确的）")
    else:
        print(f"  ✅ 大多数样本的B是可见的（{len(results) - invisible_count}/{len(results)}）")
        print(f"  → 如果模型仍然全输出NOT_VISIBLE，可能是训练问题（拒答偏置）")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_json", type=str, required=True,
                       help="测试数据question.json路径")
    parser.add_argument("--fiveframes_root", type=str, required=True,
                       help="five_frames数据根目录")
    parser.add_argument("--max_samples", type=int, default=10,
                       help="最大诊断样本数")
    
    args = parser.parse_args()
    
    results = diagnose_samples(
        Path(args.question_json),
        Path(args.fiveframes_root),
        args.max_samples
    )
    
    # 保存结果
    output_file = Path("outputs/diagnose_test_samples.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 诊断结果已保存到: {output_file}")
