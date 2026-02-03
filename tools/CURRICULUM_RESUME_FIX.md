# Curriculum Training Resume 修复说明

## 问题描述

Stage 2 和 Stage 3 没有正确 resume 训练，导致：
- Stage 2 和 Stage 3 没有开始训练
- 训练进程在初始化后停止

## 根本原因

Transformers Trainer 的 resume 逻辑：
- 只有当 `output_dir` 中已经存在 checkpoint 时，才会自动 resume
- Stage 2 的 `output_dir` 是新的目录（`Curriculum-25pct-Stage2`），没有 checkpoint
- 因此 Trainer 不会自动 resume，而是从 step 0 开始训练
- 但由于 `max_steps=60` 是相对于新开始的步数，而不是累计步数，导致训练逻辑混乱

## 修复方案

修改 `run_curriculum_3stage.py`，让 Stage 2 和 Stage 3 从上一阶段的 checkpoint 目录自动 resume：

1. **Stage 1**: 使用正常的 `output_dir`
2. **Stage 2/3**: 
   - 找到上一阶段的最新 checkpoint
   - 使用 checkpoint 目录作为 `output_dir`（这样 Trainer 会自动 resume）
   - 训练完成后，将新的 checkpoint 复制到最终的 `output_dir`

## 修复内容

### 1. 修改 resume 逻辑

```python
if stage_idx == 0:
    # Stage 1: 正常流程
    actual_output_dir = output_dir
else:
    # Stage 2/3: 从上一阶段的 checkpoint resume
    checkpoint_path = find_latest_checkpoint(prev_output)
    actual_output_dir = Path(checkpoint_path)  # 使用 checkpoint 目录作为 output_dir
```

### 2. 训练后复制 checkpoint

训练完成后，将新的 checkpoint 从 checkpoint 目录复制到最终的 output_dir：

```python
if stage_idx > 0 and actual_output_dir != output_dir:
    # 复制最新的 checkpoint 到最终的 output_dir
    copy_latest_checkpoint(actual_output_dir, output_dir)
```

## 预期行为

### Stage 1
- 从 `BASE_MODEL` 开始训练
- 训练 40 步
- 输出到 `Curriculum-25pct-Stage1/`

### Stage 2
- 从 `Curriculum-25pct-Stage1/checkpoint-40` resume
- 训练 60 步（累计到 step 100）
- 输出到 `Curriculum-25pct-Stage1/checkpoint-100`（临时）
- 然后复制到 `Curriculum-25pct-Stage2/checkpoint-100`

### Stage 3
- 从 `Curriculum-25pct-Stage2/checkpoint-100` resume
- 训练 100 步（累计到 step 200）
- 输出到 `Curriculum-25pct-Stage2/checkpoint-200`（临时）
- 然后复制到 `Curriculum-25pct-Stage3/checkpoint-200`

## 验证

运行训练脚本后，检查：

1. **Stage 1**: `trainer_state.json` 中 `global_step=40`
2. **Stage 2**: `trainer_state.json` 中 `global_step=100`
3. **Stage 3**: `trainer_state.json` 中 `global_step=200`

## 注意事项

- Stage 2 和 Stage 3 的训练会在上一阶段的 checkpoint 目录中进行
- 训练完成后，新的 checkpoint 会被复制到最终的 output_dir
- Stage 1 的 checkpoint-40 不会被覆盖（因为 Stage 2 会创建 checkpoint-100 等）
