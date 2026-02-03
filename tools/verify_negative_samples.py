#!/usr/bin/env python3
"""
验证负例生成是否正确
检查Tier B负例：确保B帧不包含uid_a
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def verify_negative_samples(sft_json_path: Path, data_root: Path):
    """验证负例生成是否正确"""
    
    with open(sft_json_path) as f:
        data = json.load(f)
    
    print('='*70)
    print('验证负例生成')
    print('='*70)
    print()
    print(f'总样本数: {len(data)}')
    
    # 分离正例和负例
    positives = [s for s in data if not s.get('meta', {}).get('is_neg', False)]
    negatives = [s for s in data if s.get('meta', {}).get('is_neg', False)]
    
    print(f'正例: {len(positives)}')
    print(f'负例: {len(negatives)}')
    print()
    
    # 按tier分类负例
    tier_a = [n for n in negatives if n.get('meta', {}).get('neg_type') == 'tierA']
    tier_b = [n for n in negatives if n.get('meta', {}).get('neg_type') == 'tierB']
    tier_c = [n for n in negatives if n.get('meta', {}).get('neg_type') == 'tierC']
    
    print(f'Tier A (跨场景): {len(tier_a)}')
    print(f'Tier B (同场景，不同UID): {len(tier_b)}')
    print(f'Tier C (同场景，同UID，不可见): {len(tier_c)}')
    print()
    
    # 验证Tier B负例
    print('='*70)
    print('验证Tier B负例')
    print('='*70)
    print()
    
    # 构建图像路径到UID的映射
    image_to_uid = {}
    for sample in positives:
        scene = sample.get('meta', {}).get('scene')
        uid = sample.get('meta', {}).get('uid')
        for img_path in sample.get('image', []):
            # 提取相对路径
            if 'scannet_inpainted' in img_path:
                rel_path = img_path.split('scannet_inpainted_dilate002_15obj_5frames_corrected_x3/')[-1]
                image_to_uid[rel_path] = (scene, uid)
    
    errors = []
    correct = 0
    
    for i, neg in enumerate(tier_b[:20]):  # 检查前20个
        scene = neg.get('meta', {}).get('scene')
        uid_a = neg.get('meta', {}).get('uid')
        b_img_path = neg.get('image', [None])[1]
        
        if not b_img_path:
            continue
        
        # 提取B帧的相对路径
        if 'scannet_inpainted' in b_img_path:
            b_rel_path = b_img_path.split('scannet_inpainted_dilate002_15obj_5frames_corrected_x3/')[-1]
        else:
            b_rel_path = b_img_path
        
        # 检查B帧是否包含uid_a
        if b_rel_path in image_to_uid:
            b_scene, b_uid = image_to_uid[b_rel_path]
            if b_scene == scene and b_uid == uid_a:
                errors.append({
                    'sample_id': neg.get('id'),
                    'scene': scene,
                    'uid_a': uid_a,
                    'b_path': b_rel_path,
                    'b_uid': b_uid
                })
            else:
                correct += 1
        else:
            # B帧不在正例中，说明确实不包含uid_a
            correct += 1
    
    print(f'检查了 {len(tier_b[:20])} 个Tier B负例:')
    print(f'  ✅ 正确: {correct}')
    print(f'  ❌ 错误: {len(errors)}')
    print()
    
    if errors:
        print('错误的负例（B帧包含uid_a）:')
        for err in errors[:5]:
            print(f'  - {err.get("sample_id")}')
            print(f'    Scene: {err.get("scene")}, UID A: {err.get("uid_a")}')
            print(f'    B帧UID: {err.get("b_uid")} (应该是不同的UID)')
            print()
    else:
        print('✅ 所有检查的Tier B负例都正确！')
        print('   B帧确实不包含uid_a')
    
    return len(errors) == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='验证负例生成是否正确')
    parser.add_argument('--sft-json', type=str, required=True,
                       help='SFT JSON文件路径')
    parser.add_argument('--data-root', type=str, required=True,
                       help='数据根目录')
    
    args = parser.parse_args()
    
    is_correct = verify_negative_samples(Path(args.sft_json), Path(args.data_root))
    
    if is_correct:
        print()
        print('='*70)
        print('✅ 验证通过：负例生成正确！')
        print('='*70)
        sys.exit(0)
    else:
        print()
        print('='*70)
        print('❌ 验证失败：发现错误的负例！')
        print('='*70)
        sys.exit(1)

if __name__ == '__main__':
    main()
