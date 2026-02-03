#!/usr/bin/env python3
"""
跨视角一致性训练脚本（使用拼接图格式，只包含正例）
"""
import subprocess
import os
import sys
from pathlib import Path

# 设置工作目录
os.chdir(Path(__file__).parent)
os.environ['PYTHONPATH'] = str(Path.cwd())

# 检查torch是否可用
def check_torch():
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        return True
    except ImportError:
        print("❌ 当前环境没有安装PyTorch")
        print("请先激活conda环境: conda activate roborefer")
        return False

# ==================== 训练配置 ====================
# Base模型
base_model = "./runs/train/RoboRefer-2B-Depth-Align"

# 选择数据版本：original 或 x3
# - original: 使用原始数据（未降采样）
# - x3: 使用降采样数据（680x440，降采样3倍）
DATA_VERSION = os.environ.get("DATA_VERSION", "x3")  # 默认使用x3

if DATA_VERSION == "original":
    data_mixture = "crossview_concat_corrected_original"
    output_dir = "runs/train/RoboRefer-2B-CrossView-Concat-Original"
    print("📂 使用原始数据拼接图（未降采样）")
elif DATA_VERSION == "x3":
    data_mixture = "crossview_concat_corrected_x3"
    output_dir = "runs/train/RoboRefer-2B-CrossView-Concat-X3"
    print("📂 使用降采样数据拼接图（680x440）")
else:
    print(f"❌ 错误: DATA_VERSION必须是 'original' 或 'x3'，当前为: {DATA_VERSION}")
    print("   设置方式: export DATA_VERSION=original 或 export DATA_VERSION=x3")
    sys.exit(1)

# 检查base model
if not os.path.exists(base_model):
    print(f"❌ Base model不存在: {base_model}")
    sys.exit(1)

print("="*70)
print("🚀 跨视角一致性训练（拼接图格式，只包含正例）")
print("="*70)
print(f"✅ Base model: {base_model}")
print(f"✅ 数据集: {data_mixture}")
print(f"✅ 输出目录: {output_dir}")
print(f"\n📋 训练配置:")
print(f"   - 数据格式: 拼接图（单张图像）")
print(f"   - 样本类型: 只包含正例，不包含负例")
print(f"   - Batch size: 1 per device")
print(f"   - Gradient accumulation: 4 steps")
print(f"   - Effective batch size: 4")
print(f"   - Epochs: 2")
print(f"   - Learning rate: 1e-5")
print(f"   - Image aspect ratio: dynamic")
print(f"   - 输出格式: [(x, y)]")
print("="*70)
print()

# 创建输出目录
Path(f"{output_dir}/model").mkdir(parents=True, exist_ok=True)

# 检查环境
if not check_torch():
    sys.exit(1)

# 检查可用的GPU
try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True
    )
    if gpu_info.returncode == 0:
        gpus = []
        for line in gpu_info.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_idx = int(parts[0])
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                mem_free = mem_total - mem_used
                gpus.append((gpu_idx, mem_free, mem_total))
        
        # 选择显存最多的GPU
        gpus.sort(key=lambda x: x[1], reverse=True)
        best_gpu = gpus[0][0]
        print(f"✅ 选择GPU {best_gpu} (可用显存: {gpus[0][1]}MB / {gpus[0][2]}MB)")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
except Exception as e:
    print(f"⚠️  GPU选择失败: {e}")

# ==================== 构建训练命令 ====================
cmd = [
    sys.executable, "-m", "torch.distributed.run",
    "--nnodes=1",
    "--nproc_per_node=1",
    "--master_port", "29513",  # 使用新的端口避免冲突
    "llava/train/train_mem.py",
    "--deepspeed", "scripts/zero3.json",
    "--model_name_or_path", base_model,
    "--chat_template", "qwen2",
    "--data_mixture", data_mixture,
    "--vision_tower", "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448",
    "--depth_tower", "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448",
    "--mm_vision_select_feature", "cls_patch",
    "--mm_projector", "mlp_downsample_3x3_fix",
    "--depth_projector", "mlp_downsample_3x3_fix",
    "--enable_depth", "False",  # 拼接图不使用depth
    "--use_depth_tower", "False",
    "--tune_vision_tower", "True",
    "--tune_mm_projector", "True",
    "--tune_language_model", "True",
    "--tune_depth_tower", "False",
    "--tune_depth_projector", "False",
    "--mm_vision_select_layer", "-2",
    "--mm_use_im_start_end", "False",
    "--mm_use_im_patch_token", "False",
    "--image_aspect_ratio", "dynamic",
    "--bf16", "True",
    "--output_dir", f"{output_dir}/model",
    "--num_train_epochs", "2",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "500",
    "--save_total_limit", "3",
    "--learning_rate", "1e-5",
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "10",
    "--model_max_length", "16384",
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "4",
    "--report_to", "none"
]

print("🚀 执行训练命令:")
print(" ".join(cmd))
print("\n" + "="*70 + "\n")

# ==================== 运行训练 ====================
try:
    subprocess.run(cmd, check=True)
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print(f"📁 模型保存在: {output_dir}/model")
    print("="*70)
except subprocess.CalledProcessError as e:
    print(f"\n❌ 训练出错 (exit code: {e.returncode})")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n⚠️  训练被用户中断")
    sys.exit(0)
