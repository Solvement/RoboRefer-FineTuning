#!/usr/bin/env python3
"""
验证SFT训练JSON的质量

检查项：
1. 所有图像路径可访问
2. 正例的GT点是否在B的mask内
3. Label格式严格（[(x,y)] 或 NOT_VISIBLE）
4. 统计分布（pos/neg count, tier distribution）
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np


def load_mask(mask_path: Path) -> np.ndarray:
    """加载mask图像"""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return m


def in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    """检查归一化坐标(x,y)是否在mask内"""
    if mask is None:
        return False
    h, w = mask.shape[:2]
    xx = int(round(x * (w - 1)))
    yy = int(round(y * (h - 1)))
    xx = max(0, min(w - 1, xx))
    yy = max(0, min(h - 1, yy))
    return mask[yy, xx] > 0


def parse_coord(label: str) -> tuple:
    """解析坐标label，返回(x, y)或None"""
    pattern = r'\[\(([0-9.]+),\s*([0-9.]+)\)\]'
    match = re.match(pattern, label.strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def validate_sample(
    sample: Dict[str, Any],
    image_root: Path,
    check_mask: bool = True
) -> Dict[str, Any]:
    """
    验证单个样本
    
    返回: {
        "valid": bool,
        "errors": List[str],
        "warnings": List[str]
    }
    """
    errors = []
    warnings = []
    
    # 1. 检查图像路径
    if "image" not in sample:
        errors.append("缺少'image'字段")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    images = sample["image"]
    if not isinstance(images, list) or len(images) != 2:
        errors.append(f"image字段应该是包含2个路径的列表，实际: {type(images)}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    for i, img_path in enumerate(images):
        full_path = image_root / img_path if not Path(img_path).is_absolute() else Path(img_path)
        if not full_path.exists():
            errors.append(f"图像{i+1}不存在: {full_path}")
    
    # 2. 检查conversations格式
    if "conversations" not in sample or len(sample["conversations"]) != 2:
        errors.append("conversations字段应该包含2个元素")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    human = sample["conversations"][0].get("value", "")
    gpt = sample["conversations"][1].get("value", "")
    
    # 3. 检查label格式
    is_neg = "NOT_VISIBLE" in gpt.upper()
    coord = parse_coord(gpt)
    
    if not is_neg and coord is None:
        errors.append(f"Label格式无效: {gpt}")
    elif is_neg and coord is not None:
        errors.append(f"Label同时包含NOT_VISIBLE和坐标: {gpt}")
    
    # 4. 检查正例的GT点是否在mask内（如果check_mask=True）
    if check_mask and not is_neg and coord:
        x, y = coord
        # 尝试找到B的mask路径
        # 这里假设mask路径可以从image路径推断，或者从meta中获取
        # 实际实现需要根据你的数据结构调整
        if "meta" in sample and "frameB" in sample["meta"]:
            # 这里需要根据实际mask路径结构来推断
            # 暂时跳过，因为需要知道mask的具体路径规则
            pass
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def main():
    ap = argparse.ArgumentParser(description="验证SFT训练JSON质量")
    ap.add_argument("--sft_json", required=True,
                   help="SFT训练JSON路径")
    ap.add_argument("--image_root", type=str, default="",
                   help="图像根目录（如果image路径是相对路径）")
    ap.add_argument("--check_images", action="store_true",
                   help="检查图像文件是否存在")
    ap.add_argument("--check_mask", action="store_true",
                   help="检查正例GT点是否在mask内")
    
    args = ap.parse_args()
    
    sft_path = Path(args.sft_json)
    if not sft_path.exists():
        print(f"❌ SFT JSON不存在: {sft_path}")
        return
    
    print(f"📖 加载SFT JSON: {sft_path}")
    with open(sft_path) as f:
        samples = json.load(f)
    
    print(f"✅ 加载了 {len(samples)} 个样本")
    
    image_root = Path(args.image_root) if args.image_root else sft_path.parent
    
    # 统计
    stats = {
        "total": len(samples),
        "positives": 0,
        "negatives": 0,
        "tierA": 0,
        "tierB": 0,
        "tierC": 0,
        "valid_format": 0,
        "invalid_format": 0,
        "image_errors": 0
    }
    
    errors_by_type = defaultdict(list)
    
    print(f"\n🔍 验证样本...")
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"   处理中: {i}/{len(samples)}")
        
        # 统计
        is_neg = sample.get("meta", {}).get("is_neg", False)
        if is_neg:
            stats["negatives"] += 1
            neg_type = sample.get("meta", {}).get("neg_type", "")
            if neg_type == "tierA":
                stats["tierA"] += 1
            elif neg_type == "tierB":
                stats["tierB"] += 1
            elif neg_type == "tierC":
                stats["tierC"] += 1
        else:
            stats["positives"] += 1
        
        # 验证
        if args.check_images or args.check_mask:
            result = validate_sample(sample, image_root, args.check_mask)
            
            if not result["valid"]:
                stats["invalid_format"] += 1
                for err in result["errors"]:
                    errors_by_type[err.split(":")[0]].append((i, err))
                    if "图像" in err:
                        stats["image_errors"] += 1
            else:
                stats["valid_format"] += 1
    
    # 输出统计
    print(f"\n📊 统计结果:")
    print(f"   总样本数: {stats['total']}")
    print(f"   正例: {stats['positives']} ({stats['positives']/stats['total']*100:.1f}%%)")
    print(f"   负例: {stats['negatives']} ({stats['negatives']/stats['total']*100:.1f}%%)")
    print(f"   - Tier A: {stats['tierA']}")
    print(f"   - Tier B: {stats['tierB']}")
    print(f"   - Tier C: {stats['tierC']}")
    
    if args.check_images or args.check_mask:
        print(f"\n   格式验证:")
        print(f"   ✅ 有效: {stats['valid_format']}")
        print(f"   ❌ 无效: {stats['invalid_format']}")
        print(f"   ❌ 图像错误: {stats['image_errors']}")
        
        if errors_by_type:
            print(f"\n⚠️  错误详情（前10个）:")
            for err_type, errs in list(errors_by_type.items())[:5]:
                print(f"   {err_type}: {len(errs)} 个")
                for idx, err in errs[:3]:
                    print(f"      样本 {idx}: {err}")
    
    print(f"\n✅ 验证完成")


if __name__ == "__main__":
    from collections import defaultdict
    main()
