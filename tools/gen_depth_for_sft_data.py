#!/usr/bin/env python3
"""
只为SFT数据中使用的图像生成深度图像（更高效）
"""
import json
import sys
from pathlib import Path
from gen_official_depth import load_depth_anything_model, generate_depth_image, generate_depths

def extract_image_paths_from_sft(sft_json_path: Path, data_root: Path):
    """从SFT JSON中提取所有需要的图像路径"""
    with open(sft_json_path) as f:
        data = json.load(f)
    
    image_paths = set()
    for sample in data:
        for img_path in sample.get('image', []):
            img_path_obj = Path(img_path)
            # 转换为相对于data_root的路径
            if str(data_root) in str(img_path_obj):
                rel_path = img_path_obj.relative_to(data_root)
            elif img_path_obj.is_absolute():
                # 尝试找到data_root在路径中的位置
                parts = img_path_obj.parts
                try:
                    idx = next(i for i, p in enumerate(parts) if 'scannet_inpainted' in p)
                    rel_path = Path(*parts[idx+1:])
                except StopIteration:
                    rel_path = img_path_obj.name
            else:
                rel_path = img_path_obj
            
            image_paths.add(data_root / rel_path)
    
    return list(image_paths)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='为SFT数据中的图像生成深度')
    parser.add_argument('--sft-json', type=str, required=True,
                       help='SFT JSON文件路径')
    parser.add_argument('--data-root', type=str, required=True,
                       help='RGB图像根目录')
    parser.add_argument('--output-root', type=str, required=True,
                       help='深度输出根目录')
    parser.add_argument('--encoder', type=str, default='vitl',
                       choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input-size', type=int, default=518)
    
    args = parser.parse_args()
    
    sft_json = Path(args.sft_json)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    print(f"📂 从SFT数据中提取图像路径: {sft_json}")
    image_paths = extract_image_paths_from_sft(sft_json, data_root)
    print(f"✅ 找到 {len(image_paths)} 张需要生成深度的图像")
    
    # 创建临时目录结构，只包含需要的图像
    # 但为了使用现有的generate_depths函数，我们需要一个包含这些图像的目录
    # 或者直接调用generate_depth_image
    
    # 加载模型
    print(f"🔧 加载Depth Anything V2模型...")
    model_info = load_depth_anything_model(
        args.encoder, args.device, args.checkpoint
    )
    
    # 生成深度
    output_root.mkdir(parents=True, exist_ok=True)
    depth_map = {}
    success_count = 0
    
    from tqdm import tqdm
    for rgb_path in tqdm(image_paths, desc="生成depth"):
        if not rgb_path.exists():
            print(f"⚠️  图像不存在: {rgb_path}")
            continue
        
        # 计算相对路径
        try:
            rel_path = rgb_path.relative_to(data_root)
        except ValueError:
            rel_path = Path(rgb_path.name)
        
        # 构造depth输出路径
        depth_path = output_root / rel_path
        depth_path = depth_path.parent / (depth_path.stem + '_depth.png')
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果已存在，跳过
        if depth_path.exists():
            depth_map[str(rel_path)] = str(depth_path.relative_to(output_root))
            success_count += 1
            continue
        
        # 生成depth
        if generate_depth_image(model_info, rgb_path, depth_path, args.device, args.input_size):
            depth_map[str(rel_path)] = str(depth_path.relative_to(output_root))
            success_count += 1
    
    # 保存映射文件
    map_file = output_root / 'depth_map.json'
    with open(map_file, 'w') as f:
        json.dump(depth_map, f, indent=2)
    
    print(f"\n✅ 完成！")
    print(f"   - 成功生成: {success_count}/{len(image_paths)} 张depth图像")
    print(f"   - 输出目录: {output_root}")
    print(f"   - 映射文件: {map_file}")

if __name__ == '__main__':
    main()
