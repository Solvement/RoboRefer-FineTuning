# RoboRefer + ScanNet 项目详细总结与分析报告

本文档从 ScanNet 数据格式、项目目标、RoboRefer 模型、历史工作、问题与解决、当前进度及后续计划等方面做**非常详细**的梳理。

---

## 一、ScanNet 数据集的格式与形式

### 1.1 原始 ScanNet 与项目中的衍生数据

本项目中使用的**不是**原始 ScanNet 的裸 RGB-D 序列，而是基于 ScanNet 加工后的 **“five_frames / x3”** 数据。数据根目录为：

- **原始分辨率**：`/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected`
- **x3 降采样**：`/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3`

含义简要说明：

- **scannet**：源自 ScanNet 室内场景。
- **inpainted**：对物体区域做了 inpainting（抠掉物体后补全背景）。
- **dilate002_15obj**：mask 做了膨胀等处理，与约 15 类物体相关。
- **5frames**：每个物体实例对应多视角下的 5 个视角（frame）。
- **corrected**：经过校正的版本。
- **x3**：在空间上做了约 1/3 的降采样，用于加快训练与评测。

### 1.2 目录结构（x3 数据）

在 `DATA_ROOT_X3` 下，结构为：

```
{split}/{scene_id}/uid_{uid}/
```

- **split**：如 `train`、`validation`。
- **scene_id**：场景 ID（如 `f3d64c30f8`）。
- **uid**：同一场景内某个物体实例的唯一 ID。

每个 `uid` 目录下包含**多视角、多类型**的图片与标注，命名规则为：

- `01_{frame}_original.png`：视角 1 的原始 RGB 图。
- `01_{frame}_inpainted.png`：视角 1 的 inpainting 图（物体区域被补全）。
- `01_{frame}_mask.png`：视角 1 的物体 mask（二值或灰度）。
- `01_{frame}_mask_dialated.png`：膨胀后的 mask（可选，用于更宽松的 GT 区域）。

其中 `01` 可替换为 `02`～`05`，表示 5 个视角；`{frame}` 为帧号（如 `008710`、`004130`）。

**总结**：每个样本 = 一个场景下的一个物体实例（uid），在最多 5 个视角下各有 original、inpainted、mask（及可选 mask_dialated），便于做跨视角匹配、缺失判断、差异定位等任务。

### 1.3 five_frames JSON 标注（部分脚本的输入）

部分数据构建脚本（如 `build_fiveframes_crossview_sft_multiimage.py`、`build_crossview_concat_sft.py`）会读取每个 uid 目录下的 **five_frames JSON**，路径形如：

`{split}/{scene_id}/uid_{uid}/{scene_id}_uid_{uid}_five_frames.json`

单条 five_frames 条目（列表中的一项）示例结构：

```json
{
  "scene_id": "f3d64c30f8",
  "uid": 129,
  "frame_id": "008710",
  "label": "bag",
  "img_w": 1920,
  "img_h": 1440,
  "x0": 11.0, "y0": 0.0, "x1": 598.0, "y1": 346.0,
  "cx_norm": 0.1585,
  "cy_norm": 0.1201,
  "original": ".../01_008710_original.png",
  "inpainted": ".../01_008710_inpainted.png",
  "mask": ".../01_008710_mask.png"
}
```

含义：

- **scene_id / uid**：场景与实例 ID。
- **frame_id**：帧号。
- **label**：物体类别（如 bag）。
- **img_w / img_h**：该帧图像宽高。
- **x0,y0,x1,y1**：2D 框（或与 bbox 相关）。
- **cx_norm, cy_norm**：归一化中心点（0～1）。
- **original / inpainted / mask**：对应图像的绝对或相对路径。

列表长度通常为 5，对应 5 个视角；下游脚本用这些信息构造「参考视角 A + 查询视角 B」的配对，以及是否可见、归一化坐标等。

### 1.4 下游任务 JSON 的约定（x3 pipeline）

由 `tools/build_spatial_tasks_x3.py` 等生成的 JSON 中：

- **image**：存**绝对路径**；可以是单张（concat）或列表（多图）。
- **prompt**：不包含 `<image>`，由 `LazySupervisedSpatialDataset` 或评测脚本根据图片数量自动在首条 human 消息前插入 `num_images` 个 `<image>\n`。
- **GT**：可见时为归一化坐标 `[(x, y)]`，不可见时为 `NOT_VISIBLE`；坐标统一相对于「query 图」宽高归一化到 [0,1]。

这些约定在训练与评测中保持一致，避免多图/拼接图时 token 与图像错位。

---

## 二、我们现在要做什么

当前目标可以概括为：

1. **在 ScanNet 衍生数据（x3）上定义并评测三类空间任务**  
   - **Task1**：跨视角对应（Cross-view correspondence）——给定参考图 A 上被标记的物体，在查询图 B 中给出同一点或回答 NOT_VISIBLE。  
   - **Task2A**：跨视角缺失（Missing across view）——仅关心「在 B 中是否可见」，输出 NOT_VISIBLE 或坐标。  
   - **Task2B**：同视角差异定位（Difference grounding）——给定 original + inpainted，在 inpainted 上指出被改变/移除的区域（一点坐标）。

2. **以 RoboRefer 为基座，做 zero-shot 评测与可选 SFT**  
   - 使用 `RoboRefer-2B-Depth-Align` 作为基座。  
   - 先做 zero-shot 评测（不训练），得到各任务的 Hit@Mask、NOT_VISIBLE 准确率、L2、Success@0.02/0.05/0.10 等指标。  
   - 再在 Task1 多图数据上做**最小配置 SFT**（如 `run_sft_t1_x3.sh`），验证能否通过少量训练提升上述能力。

3. **保证数据、训练、评测的闭环一致**  
   - 数据：x3 目录 + `build_spatial_tasks_x3.py` 生成 t1/t2a/t2b 的 train/val JSON。  
   - 训练：`LazySupervisedSpatialDataset` + `--data_path` + `--image_aspect_ratio resize`，可选冻结 vision。  
   - 评测：`eval_spatial_tasks.py` 同一套 resize、同一套坐标与 Hit@Mask 规则。

---

## 三、为什么用 RoboRefer，RoboRefer 到底是做什么的

### 3.1 为什么选 RoboRefer

- **任务匹配**：RoboRefer 面向 **spatial referring**（空间指代），能根据自然语言 + 图像输出**归一化坐标点**或判断「不可见」，与我们的 Task1/2A/2B 输出形式一致。
- **已有深度与多模态设计**：官方提供 **RoboRefer-2B-Depth-Align**，在 RefSpatial 等数据上做过 depth alignment；我们当前 x3 SFT 虽关闭 depth（`enable_depth False`），但架构上可复现或扩展带深度的实验。
- **开源与可复现**：权重、数据（RefSpatial）、评测（RefSpatial-Bench）公开，便于对比和迁移到 ScanNet 衍生数据。

因此选用 RoboRefer 作为「能输出点坐标 + NOT_VISIBLE」的 VLM 基座，在其上做 ScanNet 场景的跨视角与差异定位。

### 3.2 RoboRefer 到底是做什么的

根据 `runs/train/RoboRefer-2B-Depth-Align/README.md` 和论文信息：

- **定位**：面向机器人的 **Spatial Referring with Reasoning**——给定图像和语言描述，模型不仅做 QA，还能输出**精确的 2D 空间指代点**，用于机器人操作等。
- **能力**：  
  - 定性/定量空间问答；  
  - 输出归一化坐标 `[(x, y)]` 或 `NOT_VISIBLE`；  
  - 可与深度信息对齐（Depth-Align 版本）。
- **基座**：基于 **NVILA-Lite-2B**（Qwen2 + 视觉编码器 + 多模态投影），并在此基础上做 RefSpatial 数据上的 depth alignment 与空间指代训练。
- **输出格式**：与我们在 x3 中使用的格式一致——`[(x, y)]` 或 `NOT_VISIBLE`，便于直接对接现有评测脚本。

简言之：RoboRefer = 能「看图 + 听指令 + 输出一个点或说不可见」的 VLM，正好对应我们在 ScanNet 上的跨视角对应与差异定位需求。

---

## 四、我们想利用这个探究什么，为什么可行

### 4.1 探究目标

1. **跨视角物体对应**  
   在室内多视角下，给定参考图上一处「被标记的物体」，模型能否在另一视角图 B 中：  
   - 正确输出该物体在 B 中的位置（归一化坐标），或  
   - 正确判断 NOT_VISIBLE（遮挡、出视野等）。  
   这直接对应机器人在不同位姿下「找同一个物体」的能力。

2. **跨视角缺失判断（Task2A）**  
   仅评估「是否可见」的分类能力，可作为 Task1 的子集或简化指标。

3. **同视角差异定位（Task2B）**  
   给定 original 与 inpainted，让模型在 inpainted 上指出「被改动的区域」。  
   可视为「变化检测 / 物体移除后的 grounding」的简化版，与机器人理解场景变化相关。

4. **多图 vs 拼接图**  
   - **多图**：A、B 两张图分别输入，两个 `<image>` token。  
   - **拼接图**：左右拼接成一张图，一个 `<image>` token，prompt 中给出左图参考点在整图上的归一化坐标，要求模型在右半输出对应点。  
   探究两种输入形式对同一任务的影响，以及训练/评测管线的统一性。

### 4.2 为什么可行

- **数据已有**：x3 数据已具备多视角 original/inpainted/mask，且 GT 规则明确（bbox-center-else-random，归一化到 [0,1]），可自动生成大量 (A, B) 配对与 NOT_VISIBLE 样本。
- **模型匹配**：RoboRefer 原生支持点输出与 NOT_VISIBLE，无需改输出头，只需适配输入（多图/拼接）和 prompt。
- **评测可自动化**：Hit@Mask、L2、Success@r、NOT_VISIBLE 准确率等均可从模型输出解析后计算；concat 的坐标转换（整图 → 右半）在 `eval_spatial_tasks.py` 中已实现。
- **训练管线已打通**：`LazySupervisedSpatialDataset` 支持多图/拼接、可选的 depth、resize 与 `<image>` 插入逻辑，并与 `train_mem.py` 兼容；x3 的 SFT 脚本已验证可跑通（含 resize 与可选 FREEZE_VISION）。

因此，从数据、模型、评测、训练四方面都具备可行性。

---

## 五、之前用这个模型做了什么，为什么没继续做，遇到了什么问题，怎么解决的

### 5.1 之前做过的工作（按时间/逻辑顺序）

1. **Depth 相关**  
   - 使用 **Depth Anything V2** 等为 ScanNet 衍生数据生成深度图，并与 RGB 一起输入 RoboRefer（depth alignment 架构）。  
   - 数据管线：`gen_depth_for_sft_data.py`、`gen_official_depth.py`、`check_depth_pipeline.py` 等；训练时 `enable_depth True`，`LazySupervisedSpatialDataset` 中 image 与 depth 成对加载并保证空间一致（同 H/W）。

2. **Cross-view 多图 SFT**  
   - 从 **five_frames** 构建多图 SFT 数据：参考视角 A（带标记）+ 查询视角 B，输出 B 上的点或 NOT_VISIBLE。  
   - 脚本：`build_mv_consistency_sft.py`、`build_crossview_multimg_sft.py`、`build_fiveframes_crossview_sft_multiimage.py` 等。  
   - 数据注册在 `datasets_mixture.py`：如 `crossview_multimg_small`、`crossview_multimg_25pct`、`crossview_multimg_25pct_pos_tierA` 等，部分带 `depth_path`。

3. **Cross-view 拼接图 SFT**  
   - 将 A、B 左右拼成一张图，prompt 中给出左图参考点在整图上的坐标，模型在右半输出对应点。  
   - 脚本：`build_crossview_concat_sft.py`、`convert_multimg_to_concat_25pct.py`；shell：`generate_crossview_concat_data.sh`。  
   - 数据集：`crossview_concat_25pct`、`crossview_concat_corrected_x3` 等。

4. **Curriculum 三阶段训练**  
   - Stage1：仅正例（或简单负例），如 `crossview_multimg_25pct_pos_only` / `pos_tierA`，`max_steps=40`。  
   - Stage2/3：逐步加入更多负例或更难样本，`max_steps` 累计到 100、200。  
   - 脚本：`tools/run_curriculum_3stage.py`；文档：`CURRICULUM_FIXES_SUMMARY.md`、`CURRICULUM_RESUME_FIX.md`。

5. **训练结果与检查**  
   - Stage1 完成 40 步，loss 从约 0.96 降到约 0.78；checkpoint 完整（llm、vision_tower、depth_tower、projectors 等）。  
   - 通过 `check_training_results.py`、`verify_alignment_checklist.md` 做了通道数、空间一致、LayerNorm dtype 等检查。

### 5.2 为什么没有一直继续做下去（简要）

- **显存与稳定性**：原设置（高分辨率、dynamic aspect ratio、全参数、多 tile）易 OOM；需要降低分辨率、减少 tile、缩短序列、甚至冻结 vision。  
- **Curriculum 的 resume 与 max_steps**：Stage2/3 若 `resume_from_checkpoint` 和 `max_steps`（累计步数）配置不当，会导致未真正从 Stage1 接着训或步数不符合预期；需要显式指定 resume 和累计 max_steps。  
- **数据与评测对齐**：多种数据源（多图/拼接、25%/full、带/不带 depth）并存，评测指标和坐标约定（多图 vs concat）需统一，否则难以公平对比。  
- **目标收敛到 x3 三任务**：为简化实验与报告，后来把主线收束到「x3 数据 + Task1/2A/2B + zero-shot + 最小 SFT」，上述历史实验作为前期探索保留。

### 5.3 遇到的问题与解决方案（选列）

| 问题 | 原因/现象 | 解决方案 |
|------|-----------|----------|
| **Depth 通道数报错** | depth tensor 为 1 通道，backbone 期望 3 通道 | 在 `process_depth`、`encode_images`、`basic.py` 等处用 `_ensure_depth_3channels` 将 1 通道复制为 3 通道；在 dataset 中对 depth list 逐项保证 3 通道。 |
| **LayerNorm dtype 不一致** | bf16 下部分 LayerNorm 输出与输入 dtype 不一致 | 使用 DeepSpeed ZeRO-3 的 dtype 处理 + `patch_layer_norm_for_bf16`（`transformer_normalize_monkey_patch.py`）保证输出与输入一致。 |
| **Image/Depth 空间维度不一致** | resize/tile 后 image 与 depth 的 H/W 不一致 | 在 `LazySupervisedSpatialDataset.__getitem__` 中增加检查，确保 `processed_images` 与 `processed_depths` 的 H/W 一致，否则抛错。 |
| **Curriculum Stage2/3 未真正 resume** | `output_dir` 为新目录，Trainer 不自动 resume；max_steps 语义混淆 | Stage2/3 同时传 `--model_name_or_path` 与 `--resume_from_checkpoint`；max_steps 改为累计值（40→100→200）。详见 `CURRICULUM_FIXES_SUMMARY.md`、`CURRICULUM_RESUME_FIX.md`。 |
| **SpatialDataset 不允许 resize** | 原逻辑只允许 `dynamic` / `dynamic_s2`，用 `resize` 会触发 AssertionError | 去掉「SpatialDataset 必须 dynamic」的断言；在任何 `image_aspect_ratio` 下，只要有 image 就按 `num_images` 调用 `preprocess_rgbd` 插入 `<image>`，resize 分支沿用原有 `process_image`/stack 逻辑。 |
| **多图/拼接图 \<image\> 数量错误** | 多图应为 2 个 \<image\>，拼接图为 1 个，否则训练能跑但模型看错图 | 在 `__getitem__` 首样本做 sanity check：首条 human 消息必须以 `(DEFAULT_IMAGE_TOKEN + "\n") * num_images` 开头，否则 AssertionError。 |
| **OOM** | 原分辨率 + 长序列 + 全参数 | 使用 x3 降采样、`image_aspect_ratio resize`、`model_max_length 8192`、可选 `FREEZE_VISION=1`、`max_tiles` 减小、`torch_empty_cache_steps` 等。 |

这些修复分布在 `llava/data/dataset.py`、`llava/model/llava_arch.py`、`llava/mm_utils.py`、`llava/train/transformer_normalize_monkey_patch.py`、`tools/run_curriculum_3stage.py` 以及 SPATIAL_TASKS_X3.md 的 Error log 小节。

---

## 六、现在在做什么，什么进度，怎么做的，为什么这么做

### 6.1 当前在做的事

- **统一到 x3 三任务管线**：数据、评测、训练都围绕「x3 数据 + Task1 / Task2A / Task2B」进行。  
- **Zero-shot 评测**：用 `RoboRefer-2B-Depth-Align` 在 t1_val_multi、t1_val_concat、t2a_val_multi、t2b_val_multi 上跑评测，得到 Hit@Mask、NOT_VISIBLE acc、mean L2、Success@0.02/0.05/0.10。  
- **可选 SFT**：仅在 Task1 多图（t1_train_multi.json）上做最小 SFT（`run_sft_t1_x3.sh`），验证「少量 SFT 是否提升 Task1 指标」。

### 6.2 进度概述

- **数据**：`build_spatial_tasks_x3.py` 已实现；可在 `--data_root` x3 下生成 t1/t2a/t2b 的 train/val JSON，以及 marked/concat 图像输出（若 `emit_concat=1`）。  
- **评测**：`eval_spatial_tasks.py` 支持多图与拼接图、Hit@Mask（含 concat 的整图→右半坐标转换）、NOT_VISIBLE 与 L2；`run_zeroshot_x3.sh` 一键跑 4 个 val 并打印 summary。  
- **训练**：`run_sft_t1_x3.sh` 已跑通：`--data_mixture ""` + `--data_path tmp/spatial_tasks_x3/t1_train_multi.json`，`image_aspect_ratio resize`，`enable_depth False`，可选 `FREEZE_VISION=1`；训练日志写入 `runs/train/mv_consistency_sft_x3/train.log`。  
- **文档**：`SPATIAL_TASKS_X3.md` 记录了数据格式、任务定义、builder/eval 用法、SFT 注意事项、preprocess_rgbd 与 \<image\> 数量、data_path vs data_mixture、已知错误与修复等。

### 6.3 具体做法与设计理由

1. **数据**  
   - 使用 **x3**：降低分辨率与 I/O，加快迭代。  
   - **GT 规则**：仅用 mask 的 tight bbox，bbox 中心在 mask 内则取中心，否则在 mask 内均匀随机一点；再归一化到 [0,1]。不依赖外部检测框，可复现（seed 固定）。  
   - **绝对路径**：JSON 中 image 存绝对路径，`image_folder="/"` 即可，避免相对路径歧义。  
   - **不写 \<image\>**：prompt 中不写占位符，由 dataset/eval 按图片数插入，避免手写错误导致多图错位。

2. **训练**  
   - **LazySupervisedSpatialDataset**：当 `data_args.data_path` 非空时，`make_supervised_data_module` 直接构建该 Dataset，不再走 `data_mixture` 注册表。  
   - **resize**：与评测一致，显存更稳；多图时每张图分别 resize 再 stack。  
   - **关闭 depth**：x3 当前 SFT 不喂深度，简化管线；架构仍支持后续开启 `enable_depth`。  
   - **FREEZE_VISION**：显存紧张时冻结 vision tower，只训 LLM + mm_projector，先验证 loss 与收敛。  
   - **num_train_epochs=2、gradient_accumulation_steps=4、save_steps=500**：小规模快速试跑，便于检查过拟合与指标变化。

3. **评测**  
   - **query 图**：优先 `sample["query_original"]`，否则多图时取 `image[-1]`。  
   - **Mask 路径**：由 query 路径推断（original→mask_dialated/mask，inpainted 同理）。  
   - **Hit@Mask**：预测点映射到 query 图像素坐标，检查是否落在 mask>0；concat 时先将整图坐标转为右半 (xR,yR) 再查 mask。  
   - **输出解析**：含 `NOT_VISIBLE` 则判为不可见；否则解析第一个 `[(x, y)]` 用于 L2/Hit@Mask。

4. **One-batch 检查**  
   - `scripts/check_dataloader_t1_x3.py`：不加载完整模型，只建 dataset + dataloader，取一个 batch，检查无 assertion、image 数量与 shape 合理（如 len=2、shape [3,448,448]），推荐正式训练前跑一次。

---

## 七、后续要怎么做

### 7.1 短期（验证与收尾）

1. **跑齐 zero-shot**  
   - 执行 `bash scripts/run_zeroshot_x3.sh`，确认 4 个 val JSON 的 metrics 正常写入，并记录 Hit@Mask、NOT_VISIBLE acc、L2、Success@r 作为基线。

2. **SFT 与对比**  
   - 用 `run_sft_t1_x3.sh` 训练若干步（如 500～2000），保存 checkpoint。  
   - 用同一套 `eval_spatial_tasks.py` 对 t1_val_multi（及可选 t1_val_concat）评测，对比 zero-shot 与 SFT 后的指标；若 Hit@Mask 或 L2 明显提升，则说明 x3 Task1 数据与训练配置有效。

3. **可选：Task2A/2B 的 SFT**  
   - 若需提升 t2a/t2b，可用 `t2a_train_multi.json`、`t2b_train_multi.json` 做类似的最小 SFT（复制 `run_sft_t1_x3.sh` 改 `data_path` 与 `output_dir`）。

### 7.2 中期（扩展与稳健性）

1. **多图 + 拼接图联合**  
   - 在同一个 SFT 中混合 t1_train_multi 与 t1_train_concat（或通过 data_mixture 混合两个 JSON），观察是否比单一种输入更稳或更高。

2. **重新引入 Depth（可选）**  
   - 若 x3 有对应深度图（或由 Depth Anything 生成），可在 `LazySupervisedSpatialDataset` 中为 x3 JSON 增加 `depth` 路径，开启 `enable_depth`，复现「RGB-D」对齐对跨视角任务的增益。

3. **Curriculum 与负例**  
   - 若当前 Task1 数据只有正例，可参考之前 crossview 的 tierA/tierB 负例设计，在 x3 上构造「参考 A + 查询 B 不同物体/不同场景」的负例，做少量 curriculum（先正例后加负例）或混合比例实验。

4. **超参与稳定性**  
   - 学习率、warmup、epoch、batch 大小、是否冻结 vision 等做小范围 sweep；记录 OOM 与 loss 曲线，便于写报告或附录。

### 7.3 长期（方向与发表）

1. **与 RefSpatial / RefSpatial-Bench 对比**  
   - 在官方 benchmark 上跑同一 checkpoint，说明在 ScanNet 上的 SFT 是否损害原有能力，或做多数据源联合训练。

2. **Task2B 与「变化/移除」泛化**  
   - 将 Task2B 从「单点」扩展到框或 mask 预测（若数据有框/mask 标注），或增加更多 inpainting 策略（不同 mask 形状、多物体移除），提升差异定位的泛化。

3. **机器人仿真/实机**  
   - 将「参考图 A + 查询图 B → 点或 NOT_VISIBLE」接到导航或抓取 pipeline，做闭环评估。

4. **文档与开源**  
   - 将本报告与 SPATIAL_TASKS_X3.md、数据构建与评测命令、关键修复整理成 README 或技术报告，便于复现与后续合作。

---

## 八、关键文件与命令速查

| 用途 | 文件/命令 |
|------|-----------|
| x3 数据构建 | `tools/build_spatial_tasks_x3.py`（--data_root x3, --out_dir ./tmp/spatial_tasks_x3, --emit_concat 1） |
| Task1/2A/2B 评测 | `tools/eval_spatial_tasks.py`（--data_json, --output_json, --image_aspect_ratio resize） |
| Zero-shot 全流程 | `bash scripts/run_zeroshot_x3.sh` |
| Task1 多图 SFT | `bash scripts/run_sft_t1_x3.sh`（可选 FREEZE_VISION=1） |
| Dataloader 检查 | `PYTHONPATH=. python scripts/check_dataloader_t1_x3.py` |
| 数据集注册与 data_path | `llava/data/dataset.py`（make_supervised_data_module：data_path 非空 → LazySupervisedSpatialDataset） |
| 任务与管线说明 | `SPATIAL_TASKS_X3.md` |
| Curriculum 修复说明 | `tools/CURRICULUM_FIXES_SUMMARY.md`、`tools/CURRICULUM_RESUME_FIX.md` |
| 训练对齐检查 | `tools/verify_alignment_checklist.md`、`tools/check_training_results.py` |

---

**报告完成。** 若需对某一节（如数据格式、某次报错、或后续计划）做更细的展开或补充到代码级说明，可以指定章节或文件名继续写。
