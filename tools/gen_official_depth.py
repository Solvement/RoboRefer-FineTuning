#!/usr/bin/env python3
"""
使用RoboRefer官方的Depth Anything V2生成depth图像

输入：RGB图像根目录或图像路径列表
输出：depth图像到镜像目录结构，生成depth_map.json映射文件
"""
import argparse
import json
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 导入RoboRefer官方的Depth Anything V2
sys.path.insert(0, str(Path(__file__).parent.parent / "API" / "Depth_Anything_V2"))
from depth_anything_v2.dpt import DepthAnythingV2


def load_depth_anything_model(encoder='vitl', device='cuda', checkpoint_path=None):
    """加载RoboRefer官方的Depth Anything V2模型"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    if encoder not in model_configs:
        raise ValueError(f"Unknown encoder: {encoder}. Choose from {list(model_configs.keys())}")
    
    model = DepthAnythingV2(**model_configs[encoder])
    
    # 尝试加载checkpoint
    if checkpoint_path is None:
        # 默认checkpoint路径
        checkpoint_path = f'/home/zhouenshen/ckpt/depthanything/depth_anything_v2_{encoder}.pth'
        # 如果默认路径不存在，尝试其他常见路径
        if not Path(checkpoint_path).exists():
            alt_paths = [
                f'/local_data/cy2932/checkpoints/depth/depth_anything_v2_{encoder}.pth',
                f'checkpoints/depth_anything_v2_{encoder}.pth',
                f'./checkpoints/depth_anything_v2_{encoder}.pth',
                f'API/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth',
            ]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    checkpoint_path = alt_path
                    break
    
    if not Path(checkpoint_path).exists():
        # 尝试使用transformers库的HuggingFace模型
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   尝试使用HuggingFace transformers模型...")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            model_name_map = {
                'vits': 'LiheYoung/depth-anything-v2-small-hf',
                'vitb': 'LiheYoung/depth-anything-v2-base-hf',
                'vitl': 'LiheYoung/depth-anything-v2-large-hf',
                'vitg': 'LiheYoung/depth-anything-v2-large-hf'  # vitg fallback to large
            }
            model_name = model_name_map.get(encoder, model_name_map['vitl'])
            print(f"   使用HuggingFace模型: {model_name}")
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForDepthEstimation.from_pretrained(model_name)
            model = model.to(device).eval()
            return (processor, model, 'transformers')  # 返回tuple标识使用transformers
        except Exception as e:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Transformers fallback also failed: {e}\n"
                f"Please download Depth Anything V2 {encoder} checkpoint and specify with --checkpoint"
            )
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device).eval()
    return (model, 'official')  # 返回tuple标识使用官方实现


def generate_depth_image(model_info, rgb_path: Path, output_path: Path, device='cuda', input_size=518):
    """
    使用Depth Anything V2生成depth图像（支持官方实现或transformers）
    
    输出格式：uint16 PNG，单位mm（与RoboRefer训练格式一致）
    """
    try:
        # 读取RGB图像
        raw_image = cv2.imread(str(rgb_path))
        if raw_image is None:
            print(f"⚠️  无法读取图像: {rgb_path}")
            return False
        
        # 判断使用哪种实现
        if isinstance(model_info, tuple) and len(model_info) == 2:
            model, model_type = model_info
        else:
            # 兼容旧代码
            model = model_info
            model_type = 'official'
        
        if model_type == 'transformers':
            # 使用transformers库
            processor, model_obj = model
            from PIL import Image as PILImage
            image_pil = PILImage.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_obj(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # 转换为numpy并调整尺寸
            depth = predicted_depth.cpu().numpy()[0, 0]
            h, w = raw_image.shape[:2]
            from scipy.ndimage import zoom
            depth = zoom(depth, (h / depth.shape[0], w / depth.shape[1]))
        else:
            # 使用官方实现
            depth = model.infer_image(raw_image, input_size, device=device)
        
        # 转换为uint16，单位mm
        # Depth Anything输出的是相对深度，需要转换为绝对深度（mm）
        # 假设最大深度为20m（20000mm），这是RoboRefer常用的范围
        max_depth_mm = 20000
        depth_mm = (depth * max_depth_mm).clip(0, 65535).astype(np.uint16)
        
        # 保存为uint16 PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth_img = Image.fromarray(depth_mm, mode='I;16')
        depth_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"⚠️  生成depth失败 {rgb_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def collect_image_files(input_path: Path, extensions=None):
    """收集所有图像文件"""
    if extensions is None:
        extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    
    image_files = []
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            image_files.append(input_path)
    elif input_path.is_dir():
        for ext in extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
    
    return sorted(image_files)


def generate_depths(
    input_root: Path,
    output_root: Path,
    encoder='vitl',
    device='cuda',
    checkpoint_path=None,
    input_size=518,
    max_images=None
):
    """
    为输入目录中的所有RGB图像生成depth图像
    
    Args:
        input_root: RGB图像根目录
        output_root: depth输出根目录（镜像结构）
    """
    print(f"📂 扫描RGB图像: {input_root}")
    image_files = collect_image_files(input_root)
    print(f"✅ 找到 {len(image_files)} 张图像")
    
    # 如果指定了max_images，只处理前N张（用于测试）
    if max_images is not None and max_images > 0:
        original_count = len(image_files)
        image_files = image_files[:max_images]
        print(f"📝 测试模式: 只处理前 {len(image_files)}/{original_count} 张图像")
    
    if len(image_files) == 0:
        print("❌ 未找到任何图像文件")
        return False
    
    # 加载模型
    print(f"🔧 加载Depth Anything V2模型 (encoder={encoder})...")
    try:
        model = load_depth_anything_model(encoder, device, checkpoint_path)
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 生成depth并构建映射
    depth_map = {}
    success_count = 0
    
    for rgb_path in tqdm(image_files, desc="生成depth"):
        # 计算相对路径
        try:
            rel_path = rgb_path.relative_to(input_root)
        except ValueError:
            # 如果不在input_root下，使用绝对路径的basename
            rel_path = Path(rgb_path.name)
        
        # 构造depth输出路径（镜像结构）
        depth_path = output_root / rel_path
        # 保持相同文件名，但确保是.png格式
        depth_path = depth_path.parent / (depth_path.stem + '_depth.png')
        
        # 如果已存在，跳过
        if depth_path.exists():
            depth_map[str(rel_path)] = str(depth_path.relative_to(output_root))
            success_count += 1
            continue
        
        # 生成depth
        if generate_depth_image(model, rgb_path, depth_path, device, input_size):
            depth_map[str(rel_path)] = str(depth_path.relative_to(output_root))
            success_count += 1
    
    # 保存映射文件
    map_file = output_root / 'depth_map.json'
    with open(map_file, 'w') as f:
        json.dump(depth_map, f, indent=2)
    
    print(f"\n✅ 完成！")
    print(f"   - 成功生成: {success_count}/{len(image_files)} 张depth图像")
    print(f"   - 输出目录: {output_root}")
    print(f"   - 映射文件: {map_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='使用RoboRefer官方的Depth Anything V2生成depth图像'
    )
    parser.add_argument('--input-root', type=str, required=True,
                       help='RGB图像根目录')
    parser.add_argument('--output-root', type=str, required=True,
                       help='depth输出根目录（镜像结构）')
    parser.add_argument('--encoder', type=str, default='vitl',
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Depth Anything V2 encoder')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型checkpoint路径（如果不在默认位置）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--input-size', type=int, default=518,
                       help='输入图像尺寸')
    parser.add_argument('--max-images', type=int, default=None,
                       help='最大处理图像数量（用于测试，None表示处理全部）')
    
    args = parser.parse_args()
    
    generate_depths(
        Path(args.input_root),
        Path(args.output_root),
        args.encoder,
        args.device,
        args.checkpoint,
        args.input_size,
        args.max_images
    )


if __name__ == '__main__':
    main()
