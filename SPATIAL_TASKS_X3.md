## x3 Spatial Tasks Pipeline (Task1 / Task2A / Task2B)

**Repo root**: `/local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer`  
**x3 data root**: `/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3/`  
**Base model**: `./runs/train/RoboRefer-2B-Depth-Align`  
**Python env**: `/local_data/ky2738/envs/snpp2msg-rast`

---

### 1. Data structure (x3)

Under `DATA_ROOT_X3`:

- `{split}/{scene_id}/uid_{uid}/`
  - `01_{frame}_original.png`
  - `01_{frame}_inpainted.png`
  - `01_{frame}_mask.png`
  - `01_{frame}_mask_dialated.png`
  - ... similarly for `02..05`.

All downstream JSONs store **absolute image paths** and **prompts without `<image>`**.  
`LazySupervisedSpatialDataset` / eval scripts insert `<image>` tokens automatically based on number of images.

---

### 2. Tasks

#### Task1: Cross-view correspondence

- **Input (multi)**: `[A_marked, B_original]`
  - A = original with red overlay on mask.
  - B = query original (different view).
- **Output**:
  - visible: `[(x, y)]` normalized wrt B.
  - not visible: `NOT_VISIBLE`.
- **GT rule (query B)**（**mask_bbox_center 语义**）:
  - 仅使用 `mask>0` 的外接框（tight bbox），**不依赖外部检测框**；
  - 取 bbox 中心 `(cx, cy)`：
    - 如果 `(cx, cy)` 落在 mask 内 → 用 bbox center 作为 GT；
    - 否则从 `mask>0` 里**均匀随机**采样一个像素作为 GT 点（RNG 受 `--seed` 控制）；
  - 将该点归一化到 `[0,1]`（宽高分别归一化）；  
  - 如果 `mask>0` 为空 → 视为 `NOT_VISIBLE`。

#### Task2A: Missing across view

- Same pairs as Task1.
- Only use those where **query mask empty** (GT not visible).
- GT: `NOT_VISIBLE`. Metric: NOT_VISIBLE accuracy.

#### Task2B: Difference grounding (extra / missing proxy)

- **Input (multi)**: `[original, inpainted]` (same view).
- Target region = changed/removed region defined by mask.
- **Output**: `[(x, y)]` normalized wrt **inpainted** image.
- GT point uses同样 bbox-center-else-random 规则。
- Primary metric: Hit@Mask.

---

### 3. Data builder: `tools/build_spatial_tasks_x3.py`

**Default data root**: `/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3`  
**Output root example**: `./tmp/spatial_tasks_x3`

**CLI**:

```bash
cd /local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer
/local_data/ky2738/envs/snpp2msg-rast/bin/python tools/build_spatial_tasks_x3.py \
  --data_root /local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3 \
  --out_dir ./tmp/spatial_tasks_x3 \
  --mode anchor \
  --anchor_k 1 \
  --alpha 0.45 \
  --use_dilated_mask \
  --emit_concat 1
```

**Outputs under `out_dir`**:

- **Task1 multi**:
  - `t1_train_multi.json`
  - `t1_val_multi.json`
- **Task1 concat** (if `emit_concat=1`):
  - `t1_train_concat.json`
  - `t1_val_concat.json`
- **Task2A (missing)**:
  - `t2a_train_multi.json` (if non-empty)
  - `t2a_val_multi.json`
- **Task2B (difference grounding)**:
  - `t2b_train_multi.json` (if non-empty)
  - `t2b_val_multi.json`
- **Images**:
  - Marked: `out_dir/marked_abs/{split}/{scene_id}/uid_{uid}/*_marked.png`
  - Concat: `out_dir/concat_abs/{split}/{scene_id}/uid_{uid}/*_concat.jpg`

Important JSON fields:

- Task1 multi:
  - `"image": [ABS_A_marked, ABS_B_original]`
  - `"query_original": ABS_B_original`
  - `"conversations"[0].value`: human prompt (no `<image>`).
  - `"conversations"[1].value`: `"NOT_VISIBLE"` or `"[(x, y)]"`.
- Task1 concat:
  - `"image": ABS_concat`
  - `"query_original": ABS_B_original`
  - human prompt中会给出左半参考点在**整张 concat 图**的归一化坐标：  
    若左图内点为 `(xL, yL)`，则 concat 中 `x_full = xL * 0.5`, `y_full = yL`。  
  - 模型需在右半输出同一物体点 `(x_full, y_full)`，其中 `x_full = 0.5 + xR * 0.5`, `y_full = yR`。
- Task2B:
  - `"image": [ABS_original, ABS_inpainted]`
  - `"query_original": ABS_inpainted`
  - GT 坐标归一化 wrt inpainted。

---

### 4. Eval script: `tools/eval_spatial_tasks.py`

**CLI**（示例）：

```bash
cd /local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer
/local_data/ky2738/envs/snpp2msg-rast/bin/python tools/eval_spatial_tasks.py \
  --model_name_or_path ./runs/train/RoboRefer-2B-Depth-Align \
  --data_json ./tmp/spatial_tasks_x3/t1_val_multi.json \
  --image_folder / \
  --chat_template qwen2 \
  --image_aspect_ratio resize \
  --max_new_tokens 64 \
  --output_json ./runs/eval_spatial_x3/t1_val_multi.json
```

行为：

- **多图**：`image` 是 list → 在 prompt 前插入 `<image>\n` * len(list)。
- **拼接图**：`image` 是 string → 只插一次 `<image>\n`。
- 使用 `LlavaLlamaModel` + `qwen2` chat template，`temperature=0`、`do_sample=False`。
- 输出解析：
  - 包含 `NOT_VISIBLE` → 视为 not visible。
  - 否则解析第一组 `[(x, y)]`。

Mask / Hit@Mask 逻辑：

- 确定 query 图：
  - 优先 `sample["query_original"]`。
  - 否则如果 `"image"` 是 list 且长度 ≥2，取 `image[-1]`。
- 推断 mask 路径（优先顺序）：
  - `_original.png → _mask_dialated.png → _mask.png`
  - `_inpainted.png → _mask_dialated.png → _mask.png`
  - 否则泛化为 `.png → _mask_dialated.png/_mask.png`。
- **Hit@Mask**：
  - 多图（Task1 multi / Task2B）：
    - 预测坐标直接视为 query 图自身坐标 `(x, y) ∈ [0,1]^2`；
    - 映射到 mask 像素坐标后检查是否在 `mask>0` 内。
  - 拼接图（Task1 concat）：
    - 模型输出 `(x_full, y_full)` 相对**整张 concat 图**；
    - Hit@Mask 评估时先做坐标系转换：
      - 若 `x_full < 0.5`（点落在左半 reference）：直接视为 miss（Hit@Mask=0）；
      - 否则右半 query 局部坐标：
        - `xR = (x_full - 0.5) * 2`  
        - `yR = y_full`  
        - clamp 到 `[0,1]`；
      - 再用 `(xR, yR)` 映射到 query mask 像素坐标，检查 `mask>0`。

最终 `output_json` 结构：

- `metrics`：
  - `total`, `gt_not_visible`, `visible_gt`
  - `not_visible_accuracy`
  - `mean_l2_error`, `std_l2_error`, `median_l2_error`
  - `success_002`, `success_005`, `success_010`
  - `hit_at_mask`, `hit_at_mask_count`, `hit_at_mask_total`
- `results`：逐样本记录 `pred_text`、解析坐标、L2、Hit@Mask 等。

---

### 5. Zero-shot pipeline: `scripts/run_zeroshot_x3.sh`

**运行方式**：

```bash
cd /local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer
bash scripts/run_zeroshot_x3.sh
```

步骤：

1. 使用 `/local_data/ky2738/envs/snpp2msg-rast/bin/python`，设置 `PYTHONPATH`。  
2. 如果 `./tmp/spatial_tasks_x3/` 下 JSON 不存在，则调用 `tools/build_spatial_tasks_x3.py` 构建。  
3. 对以下四个数据集做 zero-shot 评估：
   - Task1 multi：`t1_val_multi.json → runs/eval_spatial_x3/t1_val_multi.json`
   - Task1 concat：`t1_val_concat.json → runs/eval_spatial_x3/t1_val_concat.json`
   - Task2A：`t2a_val_multi.json → runs/eval_spatial_x3/t2a_val_multi.json`
   - Task2B：`t2b_val_multi.json → runs/eval_spatial_x3/t2b_val_multi.json`
4. 用一个小 Python 脚本打印 summary：
   - 对每个任务显示：Hit@Mask, NOT_VISIBLE acc, mean L2, Success@0.02/0.05/0.10。

---

### 6. Optional SFT: `scripts/run_sft_t1_x3.sh`

**用途**：在 x3 多图 Task1 数据上，用 `llava/train/train_mem.py` 做最小配置 SFT。

保证点：

- 使用 `--data_mixture ""` + `--data_path`，触发 `LazySupervisedSpatialDataset` 分支（已在 `llava/data/dataset.py` 中 patch）。  
- `image_aspect_ratio=resize`（更稳定、更省显存）。  
- `model_max_length=8192`，比默认 16384 更轻。  
- 支持通过 `FREEZE_VISION=1` 冻结 vision tower。

**运行方式**：

建议先用最稳配置跑出 200~500 step 的 loss 再考虑解冻 vision：`FREEZE_VISION=1`、`--image_aspect_ratio resize`（脚本已默认）。

```bash
cd /local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer

# 推荐先冻结视觉塔防止 OOM，并先跑一次 one-batch 检查（见 §8）：
# PYTHONPATH=. python scripts/check_dataloader_t1_x3.py
FREEZE_VISION=1 bash scripts/run_sft_t1_x3.sh

# 若显存足够，可不冻结：
# FREEZE_VISION=0 bash scripts/run_sft_t1_x3.sh
```

训练日志会写入：

- `runs/train/mv_consistency_sft_x3/train.log`

可用：

```bash
tail -n 100 -f runs/train/mv_consistency_sft_x3/train.log
```

实时查看 loss / step。

---

### 7. preprocess_rgbd 与 &lt;image&gt; token 数量（重要）

- **调用条件**：在 `LazySupervisedSpatialDataset.__getitem__` 里，**无论 `image_aspect_ratio` 为何**（`dynamic` / `dynamic_s2` / `resize`），只要有 image 都会先根据 `num_images` 调用 `preprocess_rgbd`，在 human 首条消息前插入 `num_images` 个 `<image>\n`。这样 multi-image 样本会有 2 个 `<image>`，concat 样本会有 1 个 `<image>`，训练时 2 张图才能正确对应到 2 个 image token。
- **Sanity 校验**：取第一个样本时，会检查首条消息是否以 `(DEFAULT_IMAGE_TOKEN + "\n") * num_images` 开头；若不满足会抛 `AssertionError`（multi 应为 2，concat 应为 1），避免“训练能跑但模型看错图”的隐蔽 bug。
- **resize 与 eval 对齐**：训练和评估都使用 `--image_aspect_ratio resize`，归一化坐标定义一致；Hit@Mask 评估时用预测的 [0,1] 坐标映射到 query mask 像素，与 resize/pad 策略一致。
- **Concat 评估坐标**：concat 任务模型输出为整张 concat 图的 `(x_full, y_full)`。Hit@Mask 时已做转换：`x_full < 0.5` 视为 miss；否则 `xR = (x_full - 0.5) * 2`, `yR = y_full`，再用 `(xR, yR)` 查 query（右半）mask。见 `tools/eval_spatial_tasks.py` 中 `is_concat` 分支。

### 8. One-batch dataloader 检查（推荐训练前跑一次）

不加载模型，只初始化 dataset + dataloader，取一个 batch，确认无 assertion、batch 里 image 数量与 shape 合理：

```bash
cd /local_data/ky2738/snpp-msg/snpp-msg-conversion/scannetpp-main/RoboRefer
PYTHONPATH=. /local_data/ky2738/envs/snpp2msg-rast/bin/python scripts/check_dataloader_t1_x3.py
```

成功会打印 `OK: one-batch dataloader check passed` 以及 `media['image']: len = 2`、`first image shape: torch.Size([3, 448, 448])` 等。脚本使用 `tmp/spatial_tasks_x3/t1_train_multi.json` 和 Qwen2 tokenizer + vision tower 的 image processor。

### 9. Training data path & `data_path` vs `data_mixture`

在 `llava/data/dataset.py` 中，`make_supervised_data_module` 已修改为：

- 若 `data_args.data_path` 非空：
  - 直接构建 `LazySupervisedSpatialDataset(data_path=data_args.data_path, image_folder=data_args.image_folder, ...)`。
- 否则：
  - 走原来的 `build_dataset(data_args.data_mixture, ...)` registry。

因此所有新的 SFT/微调脚本必须传：

- `--data_mixture ""`
- `--data_path <JSON>`

---

### 10. Ongoing log / issues

后续如果在这套 x3 Spatial Tasks / Zero-shot / SFT 流水线中遇到新的问题或报错，统一记录在这里。

#### 10.1 Known environment / resource notes

- GPU: 多张 NVIDIA RTX 6000 Ada，单卡 48GB。  
- 以前在**原分辨率 + 全参数 tuning + dynamic aspect_ratio** 设置下存在 OOM 问题。  
  - 这套 x3+resize+可选冻结 vision 的配置，目的是显存更稳、适合在混用机器上跑。

#### 10.2 Error log

- **SFT 启动失败：`AssertionError: SpatialDataset do not need to resize image`**（2026-01-28）
  - **现象**：`run_sft_t1_x3.sh` 用 `--image_aspect_ratio resize` 启动训练，DataLoader 第一次取样本时在 `LazySupervisedSpatialDataset.__getitem__` 里触发断言。
  - **原因**：`llava/data/dataset.py` 中 SpatialDataset 原先写死只允许 `dynamic` 或 `dynamic_s2`，不允许 `resize`。
  - **修复**：去掉该断言；并在有 image 时**无论 aspect_ratio 为何都调用 `preprocess_rgbd`**（插入 `<image>` token），resize 分支沿用已有的 `process_image` / `torch.stack` 逻辑。  
  - **结果**：SFT 可用 `--image_aspect_ratio resize` 正常启动；后续若再报错会继续追加到本节。

