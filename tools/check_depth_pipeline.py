#!/usr/bin/env python3
"""
验证depth pipeline：检查生成的SFT JSON中的depth数据

- 加载5个随机样本
- 验证文件存在
- 加载RGB和depth图像
- 打印形状并确认depth匹配预期预处理尺寸
- 可选：可视化depth为灰度图
"""
import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path: Path, is_depth=False):
    """加载图像（RGB或depth）"""
    if not path.exists():
        return None, f"文件不存在: {path}"
    
    try:
        if is_depth:
            # depth图像是uint16 PNG
            img = Image.open(path)
            img_array = np.array(img)
            return img_array, None
        else:
            # RGB图像
            img = cv2.imread(str(path))
            if img is None:
                return None, f"无法读取图像: {path}"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, None
    except Exception as e:
        return None, f"加载失败: {e}"


def check_sample(sample: dict, image_root: Path, depth_root: Path, sample_idx: int, visualize: bool = False):
    """检查单个样本"""
    print(f"\n{'='*60}")
    print(f"样本 {sample_idx}: {sample.get('id', 'unknown')}")
    print(f"{'='*60}")
    
    # 检查基本字段
    if "image" not in sample:
        print("❌ 缺少'image'字段")
        return False
    
    if "depth" not in sample:
        print("⚠️  缺少'depth'字段（可能未启用depth）")
        return True  # 不算错误，只是没有depth
    
    images = sample["image"]
    depths = sample["depth"]
    
    if not isinstance(images, list) or len(images) != 2:
        print(f"❌ 'image'字段格式错误: 期望list[2]，得到{type(images)}")
        return False
    
    if not isinstance(depths, list) or len(depths) != 2:
        print(f"❌ 'depth'字段格式错误: 期望list[2]，得到{type(depths)}")
        return False
    
    print(f"✅ 字段格式正确")
    print(f"   image: {len(images)} 个路径")
    print(f"   depth: {len(depths)} 个路径")
    
    # 检查每个图像对
    all_ok = True
    for i, (img_path, depth_path) in enumerate(zip(images, depths)):
        print(f"\n  图像对 {i+1}:")
        print(f"    RGB: {img_path}")
        print(f"    Depth: {depth_path}")
        
        # 构造完整路径
        if Path(img_path).is_absolute():
            rgb_full = Path(img_path)
        else:
            rgb_full = image_root / img_path
        
        if Path(depth_path).is_absolute():
            depth_full = Path(depth_path)
        else:
            depth_full = depth_root / depth_path
        
        # 检查文件存在
        if not rgb_full.exists():
            print(f"   ❌ RGB文件不存在: {rgb_full}")
            all_ok = False
            continue
        
        if not depth_full.exists():
            print(f"   ❌ Depth文件不存在: {depth_full}")
            all_ok = False
            continue
        
        print(f"   ✅ 文件存在")
        
        # 加载图像
        rgb_img, rgb_err = load_image(rgb_full, is_depth=False)
        if rgb_err:
            print(f"   ❌ RGB加载失败: {rgb_err}")
            all_ok = False
            continue
        
        depth_img, depth_err = load_image(depth_full, is_depth=True)
        if depth_err:
            print(f"   ❌ Depth加载失败: {depth_err}")
            all_ok = False
            continue
        
        # 检查形状
        rgb_h, rgb_w = rgb_img.shape[:2]
        depth_h, depth_w = depth_img.shape[:2]
        
        print(f"   RGB形状: {rgb_img.shape} (H={rgb_h}, W={rgb_w})")
        print(f"   Depth形状: {depth_img.shape} (H={depth_h}, W={depth_w})")
        
        # 检查depth数据类型和范围
        print(f"   Depth数据类型: {depth_img.dtype}")
        print(f"   Depth值范围: [{depth_img.min()}, {depth_img.max()}]")
        
        # 检查尺寸匹配（depth应该与RGB尺寸相同或接近）
        if abs(rgb_h - depth_h) > 10 or abs(rgb_w - depth_w) > 10:
            print(f"   ⚠️  尺寸不匹配: RGB({rgb_h}x{rgb_w}) vs Depth({depth_h}x{depth_w})")
        else:
            print(f"   ✅ 尺寸匹配")
        
        # 可视化（如果启用）
        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(rgb_img)
            axes[0].set_title(f'RGB {i+1}')
            axes[0].axis('off')
            
            # depth可视化（归一化到0-255用于显示）
            depth_vis = (depth_img.astype(np.float32) / depth_img.max() * 255).astype(np.uint8)
            axes[1].imshow(depth_vis, cmap='gray')
            axes[1].set_title(f'Depth {i+1}')
            axes[1].axis('off')
            
            plt.tight_layout()
            vis_path = Path(f"depth_check_sample{sample_idx}_pair{i+1}.png")
            plt.savefig(vis_path)
            print(f"   💾 可视化保存到: {vis_path}")
            plt.close()
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='验证depth pipeline')
    parser.add_argument('--sft_json', type=str, required=True,
                       help='SFT训练JSON文件路径')
    parser.add_argument('--image_root', type=str, required=True,
                       help='RGB图像根目录')
    parser.add_argument('--depth_root', type=str, required=True,
                       help='Depth图像根目录')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='检查的样本数量')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化depth图像')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 加载SFT JSON
    sft_json = Path(args.sft_json)
    if not sft_json.exists():
        print(f"❌ SFT JSON文件不存在: {sft_json}")
        return
    
    print(f"📂 加载SFT JSON: {sft_json}")
    with open(sft_json) as f:
        data = json.load(f)
    
    print(f"✅ 加载了 {len(data)} 个样本")
    
    # 随机选择样本
    random.seed(args.seed)
    samples_to_check = random.sample(data, min(args.num_samples, len(data)))
    
    print(f"\n🔍 检查 {len(samples_to_check)} 个随机样本...")
    
    image_root = Path(args.image_root)
    depth_root = Path(args.depth_root)
    
    success_count = 0
    for i, sample in enumerate(samples_to_check):
        if check_sample(sample, image_root, depth_root, i+1, args.visualize):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"检查结果: {success_count}/{len(samples_to_check)} 个样本通过")
    print(f"{'='*60}")
    
    if success_count == len(samples_to_check):
        print("✅ 所有样本检查通过！")
    else:
        print("⚠️  部分样本存在问题，请检查上述输出")


if __name__ == '__main__':
    main()
