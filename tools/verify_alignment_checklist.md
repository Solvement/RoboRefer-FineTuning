# 训练配置对齐检查清单

## 使用方法

在训练过程中，检查以下4个关键点是否满足：

### ✅ 检查点 1: Image Tensor 通道数

**要求**: 进入 vision_tower 的 image tensor 必须是 3 通道

**验证方法**: 在 `llava/model/llava_arch.py` 的 `encode_images` 方法中添加：

```python
def encode_images(self, images, block_sizes=None, is_depth=False, use_depth_tower=True):
    # 检查点 1: Image tensor 通道数
    if not is_depth and isinstance(images, torch.Tensor):
        if images.dim() == 4 and images.shape[1] != 3:
            raise ValueError(f"❌ Image tensor should have 3 channels, got {images.shape[1]}. Shape: {images.shape}")
        elif images.dim() == 3 and images.shape[0] != 3:
            raise ValueError(f"❌ Image tensor should have 3 channels, got {images.shape[0]}. Shape: {images.shape}")
        print(f"✅ Image tensor OK: {images.shape} (channels={images.shape[1] if images.dim()==4 else images.shape[0]})")
    
    # ... 原有代码 ...
```

**预期输出**: 
- ✅ `Image tensor OK: torch.Size([N, 3, 448, 448])` 或 `torch.Size([T, 3, 448, 448])`
- ❌ 如果看到 `[T, 1, 448, 448]`，会触发之前的 conv 报错

---

### ✅ 检查点 2: Depth Tensor 通道数

**要求**: 进入 depth_tower 的 depth tensor 必须是 3 通道（如果 backbone 是 3ch）

**验证方法**: 在 `llava/model/llava_arch.py` 的 `encode_images` 方法中添加：

```python
def encode_images(self, images, block_sizes=None, is_depth=False, use_depth_tower=True):
    # 检查点 2: Depth tensor 通道数
    if is_depth and isinstance(images, torch.Tensor):
        if images.dim() == 4 and images.shape[1] != 3:
            raise ValueError(f"❌ Depth tensor should have 3 channels, got {images.shape[1]}. Shape: {images.shape}")
        elif images.dim() == 3 and images.shape[0] != 3:
            raise ValueError(f"❌ Depth tensor should have 3 channels, got {images.shape[0]}. Shape: {images.shape}")
        print(f"✅ Depth tensor OK: {images.shape} (channels={images.shape[1] if images.dim()==4 else images.shape[0]})")
    
    # ... 原有代码 ...
```

**预期输出**: 
- ✅ `Depth tensor OK: torch.Size([T, 3, 448, 448])`
- ❌ 如果看到 `[T, 1, 448, 448]`，会触发之前的 conv 报错

---

### ✅ 检查点 3: LayerNorm/RMSNorm dtype 一致性

**要求**: LayerNorm 前后 dtype 应该一致（输入 bf16，输出也应该是 bf16）

**验证方法**: 已经在 `llava/train/transformer_normalize_monkey_patch.py` 中通过 `patch_layer_norm_for_bf16()` 处理

**验证代码**: 在 `_layer_norm_forward_with_bf16_support` 中添加：

```python
def _layer_norm_forward_with_bf16_support(self, input):
    orig_dtype = input.dtype
    # ... 转换和计算 ...
    output = output_fp32.to(orig_dtype)
    
    # 检查点 3: dtype 一致性
    if output.dtype != orig_dtype:
        raise ValueError(f"❌ LayerNorm dtype mismatch: input={orig_dtype}, output={output.dtype}")
    
    return output
```

**预期输出**: 
- ✅ 如果使用 DeepSpeed，这个检查通常自动通过
- ✅ 如果使用我们的 patch，输出 dtype 应该与输入一致

---

### ✅ 检查点 4: Image/Depth 空间维度一致性

**要求**: 同一条样本的 image/depth 在 resize/tiling 后 H, W 必须一致

**验证方法**: 在 `llava/data/dataset.py` 的 `LazySupervisedSpatialDataset.__getitem__` 中添加：

```python
def __getitem__(self, i):
    # ... 原有代码处理 image 和 depth ...
    
    # 检查点 4: 空间维度一致性
    if self.enable_depth and "image" in self.list_data_dict[i]:
        if processed_images is not None and processed_depths is not None:
            img_shape = processed_images.shape
            depth_shape = processed_depths.shape
            
            # 提取空间维度
            if img_shape.dim() == 4:
                img_h, img_w = img_shape[2], img_shape[3]
            else:
                img_h, img_w = img_shape[1], img_shape[2]
            
            if depth_shape.dim() == 4:
                depth_h, depth_w = depth_shape[2], depth_shape[3]
            else:
                depth_h, depth_w = depth_shape[1], depth_shape[2]
            
            if img_h != depth_h or img_w != depth_w:
                raise ValueError(
                    f"❌ Spatial mismatch - image: ({img_h}, {img_w}), "
                    f"depth: ({depth_h}, {depth_w})"
                )
            print(f"✅ Spatial consistent: image={img_shape}, depth={depth_shape}")
    
    # ... 原有代码 ...
```

**预期输出**: 
- ✅ `Spatial consistent: image=torch.Size([T, 3, 448, 448]), depth=torch.Size([T, 3, 448, 448])`
- ❌ 如果 H, W 不一致，会导致后续 concat 或 projector 对齐失败

---

## 快速验证命令

运行训练时，在第一个 batch 应该看到：

```
✅ Image tensor OK: torch.Size([208, 3, 448, 448]) (channels=3)
✅ Depth tensor OK: torch.Size([208, 3, 448, 448]) (channels=3)
✅ Spatial consistent: image=torch.Size([208, 3, 448, 448]), depth=torch.Size([208, 3, 448, 448])
```

如果看到任何 ❌，说明配置未对齐，需要修复。

---

## 当前状态

根据我们的修复：

1. ✅ **检查点 1 & 2**: 已通过 `_ensure_depth_3channels` 函数在所有关键位置修复
2. ✅ **检查点 3**: 已通过 `patch_layer_norm_for_bf16` 和 DeepSpeed 处理
3. ✅ **检查点 4**: 已在 dataset 中添加检查（如果 image 和 depth 都处理，应该一致）

## 验证脚本

运行训练时，这些检查会自动执行。如果训练能正常进行（没有 dtype 或通道数错误），说明所有检查都通过了。
