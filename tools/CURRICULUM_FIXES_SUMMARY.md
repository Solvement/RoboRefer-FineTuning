# Curriculum 训练脚本修复总结

## ✅ 已修复的问题

### 1. **max_steps 语义修复（关键修复）**
**问题**: Stage 2/3 使用相对步数（60/100），但 HF Trainer 的 `max_steps` 是累计步数上限。

**修复**:
- Stage 1: `max_steps=40` (0 -> 40)
- Stage 2: `max_steps=100` (40 -> 100, 额外 60 步) ✅ 从 60 改为 100
- Stage 3: `max_steps=200` (100 -> 200, 额外 100 步) ✅ 从 100 改为 200

### 2. **Resume 逻辑修复**
**问题**: Stage 2/3 只使用 `model_name_or_path`，没有显式指定 `resume_from_checkpoint`，导致 optimizer/scheduler/global_step 未恢复。

**修复**:
- Stage 2/3 现在同时使用：
  - `--model_name_or_path`: 指向上一阶段的 output_dir（加载模型权重）
  - `--resume_from_checkpoint`: 指向上一阶段的 checkpoint 目录（恢复训练状态）

### 3. **日志捕获修复**
**问题**: 只捕获了 stdout，stderr 中的错误信息丢失。

**修复**:
- 使用 `subprocess.run(..., stderr=subprocess.STDOUT)` 合并 stderr 到 stdout
- 保存完整日志到 `{output_dir}/stage_{name}_full.log`
- 添加环境变量 `PYTHONFAULTHANDLER=1` 和 `TORCH_SHOW_CPP_STACKTRACES=1` 以获取更详细的错误信息

### 4. **语法错误修复**
**问题**: `cmd.extend([...])` 缺少闭合括号 `)`。

**修复**: 在第265行的 `]` 后添加了 `)`。

### 5. **内存优化（已应用）**
- Stage 1: `max_tiles=6`, `dataloader_workers=4`
- Stage 2/3: `max_tiles=4`, `dataloader_workers=2`
- `torch_empty_cache_steps=1` (更频繁的缓存清理)

---

## 📋 修复后的配置

### Stage 1
- **Max Steps**: 40 (累计)
- **Resume**: 从 `BASE_MODEL` 开始
- **Output**: `Curriculum-25pct-Stage1/`

### Stage 2
- **Max Steps**: 100 (累计，从 40 继续)
- **Resume**: 
  - `model_name_or_path`: `Curriculum-25pct-Stage1/`
  - `resume_from_checkpoint`: `Curriculum-25pct-Stage1/checkpoint-40`
- **Output**: `Curriculum-25pct-Stage2/`

### Stage 3
- **Max Steps**: 200 (累计，从 100 继续)
- **Resume**: 
  - `model_name_or_path`: `Curriculum-25pct-Stage2/`
  - `resume_from_checkpoint`: `Curriculum-25pct-Stage2/checkpoint-100`
- **Output**: `Curriculum-25pct-Stage3/`

---

## 🚀 运行训练

现在可以运行修复后的训练脚本：

```bash
cd /local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer
python3 tools/run_curriculum_3stage.py
```

或者手动运行 Stage 2（用于调试）：

```bash
cd /local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer

export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

/local_data/ky2738/envs/snpp2msg-rast/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=1 --master_port 29521 \
  llava/train/train_mem.py \
  --deepspeed scripts/zero3.json \
  --model_name_or_path runs/train/Curriculum-25pct-Stage1 \
  --resume_from_checkpoint runs/train/Curriculum-25pct-Stage1/checkpoint-40 \
  --data_mixture crossview_multimg_25pct_pos_tierA \
  --output_dir runs/train/Curriculum-25pct-Stage2 \
  --max_steps 100 \
  --max_tiles 4 \
  --dataloader_num_workers 2 \
  --torch_empty_cache_steps 1 \
  --bf16 True \
  2>&1 | tee runs/train/Curriculum-25pct-Stage2/manual_stage2_full.log
```

---

## ✅ 验证检查点

训练完成后，检查：

1. **Stage 1**: `trainer_state.json` 中 `global_step=40`
2. **Stage 2**: `trainer_state.json` 中 `global_step=100`
3. **Stage 3**: `trainer_state.json` 中 `global_step=200`

如果步数不匹配，检查对应的 `stage_{name}_full.log` 文件以查看详细错误信息。
