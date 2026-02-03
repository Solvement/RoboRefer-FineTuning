#!/usr/bin/env python3
"""
使用Depth Anything生成depth图像

输入：RGB图像路径
输出：depth图像（PNG格式，uint16）
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    HAS_DEPTH_ANYTHING = True
except ImportError:
    HAS_DEPTH_ANYTHING = False
    print("⚠️  depth_anything_v2未安装，将尝试使用transformers库")

def load_depth_anything_model(device='cuda'):
    """加载Depth Anything模型"""
    if HAS_DEPTH_ANYTHING:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [192, 384, 768, 1536]}
        }
        model = DepthAnythingV2(**model_configs['vitl'])
        model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
        model.to(device).eval()
        return model
    else:
        # 尝试使用transformers库
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-v2-base-hf")
            model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-v2-base-hf")
            model.to(device).eval()
            return (processor, model)
        except Exception as e:
            print(f"❌ 无法加载Depth Anything模型: {e}")
            return None

def generate_depth_with_depth_anything(image_path: Path, output_path: Path, model, device='cuda'):
    """使用Depth Anything生成depth图像"""
    try:
        # 加载RGB图像
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        if HAS_DEPTH_ANYTHING:
            # 使用depth_anything_v2库
            depth = model.infer_image(image_np)
            # 转换为uint16（单位：mm，范围0-65535）
            depth_uint16 = (depth * 1000).clip(0, 65535).astype(np.uint16)
        else:
            # 使用transformers库
            processor, model_obj = model
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_obj(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # 转换为numpy并调整尺寸
            depth = predicted_depth.cpu().numpy()[0, 0]
            # 调整到原图尺寸
            from scipy.ndimage import zoom
            h, w = image_np.shape[:2]
            depth = zoom(depth, (h / depth.shape[0], w / depth.shape[1]))
            # 转换为uint16（单位：mm）
            depth_uint16 = (depth * 1000).clip(0, 65535).astype(np.uint16)
        
        # 保存depth图像
        depth_img = Image.fromarray(depth_uint16, mode='I;16')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth_img.save(output_path)
        return True
    except Exception as e:
        print(f"⚠️  生成depth失败 {image_path}: {e}")
        return False

def generate_depths_for_five_frames(five_frames_root: Path, depth_output_root: Path, device='cuda'):
    """为five_frames数据生成depth图像"""
    print(f"📂 扫描five_frames数据: {five_frames_root}")
    
    # 加载模型
    print("🔧 加载Depth Anything模型...")
    model = load_depth_anything_model(device)
    if model is None:
        print("❌ 无法加载Depth Anything模型")
        return False
    
    print("✅ 模型加载完成")
    
    # 扫描所有图像
    image_files = []
    for split in ["train", "validation"]:
        split_dir = five_frames_root / split
        if not split_dir.exists():
            continue
        
        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            
            for uid_dir in scene_dir.iterdir():
                if not uid_dir.is_dir() or not uid_dir.name.startswith("uid_"):
                    continue
                
                # 查找所有original图像
                for img_file in uid_dir.glob("*_original.png"):
                    image_files.append(img_file)
    
    print(f"✅ 找到 {len(image_files)} 张图像")
    
    # 生成depth
    success_count = 0
    for img_path in tqdm(image_files, desc="生成depth"):
        # 构造depth输出路径
        # 例如: train/scene/uid_xxx/01_001640_original.png -> depth_output/train/scene/uid_xxx/01_001640_depth.png
        rel_path = img_path.relative_to(five_frames_root)
        depth_path = depth_output_root / rel_path.parent / rel_path.name.replace("_original.png", "_depth.png")
        
        # 如果已存在，跳过
        if depth_path.exists():
            continue
        
        if generate_depth_with_depth_anything(img_path, depth_path, model, device):
            success_count += 1
    
    print(f"\n✅ 完成！成功生成 {success_count}/{len(image_files)} 张depth图像")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--five_frames_root", type=str, required=True,
                       help="five_frames数据根目录")
    parser.add_argument("--depth_output_root", type=str, required=True,
                       help="depth输出根目录")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    generate_depths_for_five_frames(
        Path(args.five_frames_root),
        Path(args.depth_output_root),
        args.device
    )
