#!/usr/bin/env python3
"""检查训练结果的完整性"""
import os
import json
from pathlib import Path

def check_checkpoint(checkpoint_dir):
    """检查 checkpoint 完整性"""
    checkpoint_dir = Path(checkpoint_dir)
    required = {
        'llm': '目录',
        'vision_tower': '目录',
        'depth_tower': '目录',
        'mm_projector': '目录',
        'depth_projector': '目录',
        'trainer_state.json': '文件',
        'config.json': '文件'
    }
    
    print(f"\n=== Checkpoint 完整性检查: {checkpoint_dir.name} ===")
    missing = []
    total_size = 0
    
    for item, item_type in required.items():
        path = checkpoint_dir / item
        if path.exists():
            if item_type == '目录':
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            else:
                size = path.stat().st_size if path.is_file() else 0
            
            total_size += size
            status = "✅" if size > 0 else "⚠️"
            print(f"{status} {item}: {size/1024/1024:.2f} MB")
        else:
            print(f"❌ {item}: 缺失")
            missing.append(item)
    
    print(f"\n总计: {len(required)-len(missing)}/{len(required)} 组件存在")
    print(f"总大小: {total_size/1024/1024/1024:.2f} GB")
    
    return len(missing) == 0, missing

def check_trainer_state(state_file):
    """检查训练状态"""
    with open(state_file) as f:
        state = json.load(f)
    
    print(f"\n=== 训练状态 ===")
    print(f"Global Step: {state['global_step']}/{state['max_steps']}")
    print(f"Epoch: {state['epoch']:.4f}")
    
    # 提取 loss 历史
    losses = [x['loss'] for x in state['log_history'] if 'loss' in x]
    if losses:
        print(f"初始 Loss: {losses[0]:.4f}")
        print(f"最终 Loss: {losses[-1]:.4f}")
        print(f"Loss 下降: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    
    # 检查训练完成度
    if state['global_step'] >= state['max_steps']:
        print("✅ 训练已完成")
        return True
    else:
        print(f"⚠️  训练未完成 ({state['global_step']}/{state['max_steps']})")
        return False

def main():
    base_dir = Path("/local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer/runs/train")
    
    # 检查 Stage 1
    stage1_dir = base_dir / "Curriculum-25pct-Stage1"
    stage1_checkpoint = stage1_dir / "checkpoint-40"
    stage1_state = stage1_dir / "trainer_state.json"
    
    print("=" * 60)
    print("训练结果检查报告")
    print("=" * 60)
    
    if stage1_checkpoint.exists():
        checkpoint_ok, missing = check_checkpoint(stage1_checkpoint)
        if stage1_state.exists():
            training_ok = check_trainer_state(stage1_state)
        else:
            print("\n⚠️  trainer_state.json 不存在")
            training_ok = False
    else:
        print("\n❌ Stage 1 checkpoint 不存在")
        checkpoint_ok = False
        training_ok = False
    
    # 检查 Stage 2
    stage2_dir = base_dir / "Curriculum-25pct-Stage2"
    print(f"\n=== Stage 2 状态 ===")
    if stage2_dir.exists():
        stage2_files = list(stage2_dir.iterdir())
        if stage2_files:
            print(f"✅ Stage 2 目录存在，包含 {len(stage2_files)} 个文件/目录")
        else:
            print("⚠️  Stage 2 目录存在但为空（可能尚未开始）")
    else:
        print("⚠️  Stage 2 目录不存在（可能尚未开始）")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"Stage 1 Checkpoint: {'✅ 完整' if checkpoint_ok else '❌ 不完整'}")
    print(f"Stage 1 训练: {'✅ 完成' if training_ok else '❌ 未完成'}")
    if missing:
        print(f"缺失组件: {', '.join(missing)}")

if __name__ == "__main__":
    main()
