#!/usr/bin/env python3
"""
为现有的SFT数据添加depth字段
"""
import json
import sys
from pathlib import Path

def get_depth_path(image_path: Path, depth_root: Path, depth_map: dict = None):
    """
    根据RGB图像路径推断depth路径
    
    Args:
        image_path: RGB图像路径（可以是绝对路径或相对路径）
        depth_root: depth图像根目录
        depth_map: depth映射字典 {rgb_rel_path: depth_rel_path}
    
    Returns:
        depth相对路径（相对于depth_root），如果找不到返回None
    """
    # 如果提供了depth_map，优先使用
    if depth_map is not None:
        # 尝试找到image_path在depth_map中的key
        # image_path可能是绝对路径，需要转换为相对路径
        for rgb_key, depth_rel in depth_map.items():
            if rgb_key in str(image_path) or str(image_path).endswith(rgb_key):
                return depth_rel
    
    # 如果没有depth_map，尝试从image_path推断
    if depth_root is None:
        # 尝试从image_path推断depth路径
        # 例如: .../01_001640_original.png -> .../01_001640_original_depth.png
        depth_path = image_path.parent / image_path.name.replace("_original.png", "_original_depth.png")
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
            
            # 构造depth路径：将_original.png替换为_original_depth.png
            depth_rel = rel_path.parent / rel_path.name.replace("_original.png", "_original_depth.png")
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

def update_sft_data_with_depth(sft_json_path: Path, depth_root: Path, output_path: Path):
    """为SFT数据添加depth字段"""
    
    # 加载depth_map
    depth_map_file = depth_root / 'depth_map.json'
    depth_map = {}
    if depth_map_file.exists():
        with open(depth_map_file) as f:
            depth_map = json.load(f)
        print(f"✅ 加载depth映射文件: {len(depth_map)} 条映射")
    else:
        print(f"⚠️  depth_map.json不存在，将使用路径推断")
    
    # 加载SFT数据
    with open(sft_json_path) as f:
        data = json.load(f)
    
    print(f"📂 加载SFT数据: {len(data)} 条样本")
    
    updated_count = 0
    missing_depth_count = 0
    
    for sample in data:
        image_list = sample.get('image', [])
        if not image_list:
            continue
        
        # 为每个图像生成depth路径
        depth_list = []
        for img_path in image_list:
            img_path_obj = Path(img_path)
            
            # 直接使用路径推断（因为深度图像应该已经生成）
            # 从绝对路径提取相对路径：validation/scene/uid/file
            img_str = str(img_path_obj)
            if 'scannet_inpainted' in img_str:
                # 提取相对路径部分
                parts = img_str.split('scannet_inpainted_dilate002_15obj_5frames_corrected_x3/')
                if len(parts) > 1:
                    rel_path_str = parts[1]
                    # 将 _original.png 替换为 _original_depth.png（深度图像命名规则）
                    depth_rel_str = rel_path_str.replace("_original.png", "_original_depth.png")
                    # 检查文件是否存在
                    depth_full_path = depth_root / depth_rel_str
                    if depth_full_path.exists():
                        depth_list.append(depth_rel_str)
                    else:
                        depth_list.append("")
                else:
                    depth_list.append("")
            else:
                # 尝试使用get_depth_path函数
                depth_path = get_depth_path(img_path_obj, depth_root, depth_map)
                if depth_path:
                    depth_list.append(depth_path)
                else:
                    depth_list.append("")
        
        # 如果至少有一个depth路径，添加depth字段
        if any(depth_list):
            sample['depth'] = depth_list
            updated_count += 1
        else:
            missing_depth_count += 1
    
    # 保存更新后的数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ 完成！")
    print(f"   - 总样本数: {len(data)}")
    print(f"   - 已添加depth: {updated_count}")
    print(f"   - 缺少depth: {missing_depth_count}")
    print(f"   - 输出文件: {output_path}")
    
    return updated_count, missing_depth_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='为SFT数据添加depth字段')
    parser.add_argument('--sft-json', type=str, required=True,
                       help='输入SFT JSON文件路径')
    parser.add_argument('--depth-root', type=str, required=True,
                       help='深度图像根目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出SFT JSON文件路径')
    
    args = parser.parse_args()
    
    update_sft_data_with_depth(
        Path(args.sft_json),
        Path(args.depth_root),
        Path(args.output)
    )

if __name__ == '__main__':
    main()
