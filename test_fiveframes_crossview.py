#!/usr/bin/env python3
"""
使用训练后的 five_frames multi-image 模型在 CrossView benchmark 上测试跨视角识别

这个脚本会：
1. 加载训练后的模型
2. 从 five_frames 数据中提取 A 和 B 图（基于 question.json 中的 scene_id, frame_a_id, frame_b_id）
3. 使用多图输入进行推理
4. 保存结果并计算指标
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在 RoboRefer 目录下运行此脚本")
    sys.exit(1)


def find_fiveframes_image(
    fiveframes_root: Path,
    scene_id: str,
    uid: str,
    frame_id: str,
    allow_fallback: bool = False
) -> Optional[Path]:
    """
    在 five_frames 数据中查找对应的 original 图像
    
    Args:
        fiveframes_root: five_frames 数据根目录
        scene_id: 场景ID（例如 "00777c41d4"）
        uid: 实例ID（例如 "128"）
        frame_id: 帧ID（例如 "001640"）
        allow_fallback: 如果找不到指定frame_id，是否允许从five_frames.json中随机选择一个
    
    Returns:
        图像路径，如果找不到返回 None
    """
    # 尝试 train 和 validation 两个目录
    for split in ["train", "validation"]:
        # 构造路径：split/scene_id/uid_xxx/XX_frame_id_original.png
        uid_dir = fiveframes_root / split / scene_id / f"uid_{uid}"
        if not uid_dir.exists():
            continue
        
        # 查找匹配的 original 图像
        for img_file in uid_dir.glob(f"*_{frame_id}_original.png"):
            return img_file
        
        # 如果没找到，尝试查找 five_frames.json 来确定文件名格式
        json_file = uid_dir / f"{scene_id}_uid_{uid}_five_frames.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    frames_data = json.load(f)
                for frame_data in frames_data:
                    if str(frame_data.get("frame_id", "")) == str(frame_id):
                        original_path = Path(frame_data.get("original", ""))
                        if original_path.exists():
                            return original_path
                        # 尝试相对路径
                        rel_path = uid_dir / original_path.name
                        if rel_path.exists():
                            return rel_path
                
                # 如果allow_fallback=True且没找到指定frame_id，随机选择一个可用的frame
                if allow_fallback and len(frames_data) > 0:
                    import random
                    fallback_frame = random.choice(frames_data)
                    original_path = Path(fallback_frame.get("original", ""))
                    if original_path.exists():
                        return original_path
                    # 尝试相对路径
                    rel_path = uid_dir / original_path.name
                    if rel_path.exists():
                        return rel_path
            except Exception as e:
                print(f"⚠️  读取 {json_file} 失败: {e}")
                continue
    
    return None


def load_model(model_path: str, device: str = "cuda"):
    """加载训练后的模型"""
    print(f"📦 加载模型: {model_path}")
    
    model_name = get_model_name_from_path(model_path)
    if not model_name or model_name == "":
        # 如果获取失败，使用默认值
        model_name = "roborefer"
        print(f"⚠️  无法从路径获取model_name，使用默认值: {model_name}")
    
    print(f"📝 Model name: {model_name}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_name, None, device_map=device, load_8bit=False, load_4bit=False
    )
    
    print(f"✅ 模型加载完成")
    return tokenizer, model, image_processor


def inference_multi_image(
    model,
    tokenizer,
    image_processor,
    image_a_path: Path,
    image_b_path: Path,
    prompt: str,
    device: str = "cuda"
) -> str:
    """
    使用多图输入进行推理
    
    Args:
        model: 加载的模型
        tokenizer: tokenizer
        image_processor: 图像处理器
        image_a_path: A 图路径
        image_b_path: B 图路径
        prompt: 提示词
        device: 设备
    
    Returns:
        模型输出文本
    """
    # 加载图像
    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    
    # 处理图像 - 直接使用 image_processor
    # 为每张图添加 batch 维度，然后 stack
    processed_a = image_processor.preprocess(image_a, return_tensors="pt")["pixel_values"][0]  # [C, H, W]
    processed_b = image_processor.preprocess(image_b, return_tensors="pt")["pixel_values"][0]  # [C, H, W]
    # 使用 half() 转换为 float16（与训练时一致）
    processed_a = processed_a.half()
    processed_b = processed_b.half()
    image_tensors = torch.stack([processed_a, processed_b], dim=0)  # [2, C, H, W]
    
    # 处理 prompt（添加图像token，两张图需要两个token）
    if DEFAULT_IMAGE_TOKEN not in prompt:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_IMAGE_TOKEN}\n" + prompt
    else:
        # 如果已经有token，确保有两个（对应两张图）
        token_count = prompt.count(DEFAULT_IMAGE_TOKEN)
        if token_count < 2:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n" * (2 - token_count) + prompt
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    
    # 推理 - 使用 media 参数（格式：Dict[str, List[torch.Tensor]]）
    # 将 [2, C, H, W] 的 tensor 拆分成两个 [C, H, W] 的 tensor
    # 确保在正确的设备上，并使用half精度（与训练时一致）
    image_tensors_list = [
        image_tensors[i].to(device=device, dtype=torch.float16) 
        for i in range(image_tensors.shape[0])
    ]
    media = {"image": image_tensors_list}
    media_config = {"image": {}}
    
    # 确保模型在eval模式
    model.eval()
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            media=media,
            media_config=media_config,
            do_sample=False,
            temperature=None,
            top_p=None,
            num_beams=1,
            max_new_tokens=100,  # 增加到100，确保能生成完整的坐标格式
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,  # 设置pad token
        )
    
    # 解码输出
    raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # 移除输入部分
    if prompt in raw_output:
        output = raw_output.replace(prompt, "").strip()
    else:
        output = raw_output
    
    # 保存原始输出用于调试
    import os
    if os.environ.get("DEBUG_OUTPUT", "0") == "1":
        print(f"DEBUG: 原始输出: {repr(output)}")
        print(f"DEBUG: 输出长度: {len(output)}")
    
    # 后处理：提取有效的坐标格式或NOT_VISIBLE
    processed_output = extract_valid_output(output)
    
    # 如果处理后输出不完整，返回原始输出的一部分用于调试
    if processed_output == output[:100] and len(output) > 20:
        # 说明extract_valid_output没有找到有效格式，返回原始输出用于分析
        return output[:200]  # 返回前200字符用于分析
    
    return processed_output


def extract_valid_output(text: str) -> str:
    """
    从模型输出中提取有效的坐标格式或NOT_VISIBLE
    
    Args:
        text: 模型原始输出
    
    Returns:
        提取后的有效输出：[(x, y)] 或 NOT_VISIBLE
    """
    import re
    
    # 先检查是否有NOT_VISIBLE
    if "NOT_VISIBLE" in text.upper():
        return "NOT_VISIBLE"
    
    # 尝试提取第一个完整的坐标格式 [(x, y)]
    # 匹配格式：[(数字, 数字)] 或 [(数字,数字)]
    pattern = r'\[\(([0-9.]+),\s*([0-9.]+)\)\]'
    match = re.search(pattern, text)
    if match:
        x, y = match.groups()
        # 验证坐标范围 [0, 1]
        try:
            x_val = float(x)
            y_val = float(y)
            if 0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0:
                return f"[({x}, {y})]"
        except ValueError:
            pass
    
    # 如果没有找到完整格式，尝试从开始部分提取
    # 模型输出通常是 [(0.100[([([... 这种格式
    # 我们需要提取 [(0.100 后面的数字
    # 先找到 [(数字 的模式
    pattern_start = r'\[\(([0-9.]+)'
    match_start = re.search(pattern_start, text)
    if match_start:
        first_num = match_start.group(1)
        # 然后尝试找到第二个数字（可能在后面）
        # 由于格式是 [(0.100[(0. 或 [(0.100[([([，我们需要更灵活的匹配
        # 尝试匹配 [(数字, 数字 或 [(数字[(数字
        pattern_two = r'\[\(([0-9.]+)[,\[\(]([0-9.]+)'
        match_two = re.search(pattern_two, text)
        if match_two:
            x, y = match_two.groups()
            try:
                x_val = float(x)
                y_val = float(y)
                if 0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0:
                    return f"[({x}, {y})]"
            except ValueError:
                pass
        
        # 如果只找到一个数字，尝试从重复的括号中提取第二个
        # 例如：[(0.100[(0. 这种情况
        pattern_second = r'\[\(([0-9.]+)\)?\[\(([0-9.]+)'
        match_second = re.search(pattern_second, text)
        if match_second:
            x, y = match_second.groups()
            try:
                x_val = float(x)
                y_val = float(y)
                if 0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0:
                    return f"[({x}, {y})]"
            except ValueError:
                pass
    
    # 如果都没有找到，尝试更宽松的匹配：找到两个数字
    # 匹配格式：数字.数字（例如 0.100）
    numbers = re.findall(r'([0-9]\.[0-9]+)', text)
    if len(numbers) >= 2:
        try:
            x_val = float(numbers[0])
            y_val = float(numbers[1])
            if 0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0:
                return f"[({numbers[0]}, {numbers[1]})]"
        except (ValueError, IndexError):
            pass
    
    # 如果只找到一个数字，尝试从输出开始部分提取
    # 模型输出通常是 [(0.100[([... 或 [(0.100, 0.)
    # 尝试匹配 [(数字, 数字) 或 [(数字,数字
    pattern_incomplete = r'\[\(([0-9.]+),\s*([0-9.]+)\)'
    match_incomplete = re.search(pattern_incomplete, text)
    if match_incomplete:
        x, y = match_incomplete.groups()
        try:
            x_val = float(x)
            # y可能是 "0." 这种不完整格式，需要处理
            if y.endswith('.'):
                # 如果y以.结尾，可能是0.0，尝试补全
                y = y.rstrip('.') + '0'
            y_val = float(y)
            if 0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0:
                return f"[({x}, {y})]"
        except ValueError:
            pass
    
    # 如果都没有找到，返回原始输出（用于调试）
    return text[:100]  # 只返回前100字符，避免太长


def test_crossview_benchmark(
    model_path: str,
    question_json: Path,
    fiveframes_root: Path,
    output_json: Path,
    device: str = "cuda",
    max_samples: Optional[int] = None
):
    """
    在 CrossView benchmark 上测试模型
    
    Args:
        model_path: 训练后的模型路径
        question_json: CrossView question.json 路径
        fiveframes_root: five_frames 数据根目录
        output_json: 输出结果 JSON 路径
        device: 设备
        max_samples: 最大测试样本数（None 表示全部）
    """
    print(f"📖 加载问题集: {question_json}")
    with open(question_json, 'r') as f:
        questions = json.load(f)
    
    if max_samples:
        questions = questions[:max_samples]
        print(f"⚠️  限制测试样本数: {max_samples}")
    
    print(f"✅ 共 {len(questions)} 个测试样本")
    
    # 加载模型
    tokenizer, model, image_processor = load_model(model_path, device)
    
    # 准备输出
    results = []
    failed = 0
    
    # 构造多图 cross-view prompt（与训练时一致）
    def build_multiimage_prompt(label: str) -> str:
        return (
            "You are given TWO separate images:\n"
            "- Image A (REFERENCE): the target object is highlighted (marked) in the image.\n"
            "- Image B (QUERY): you need to find the SAME object as in Image A.\n\n"
            f"The target in Image A is a \"{label}\". It is visually marked, so you can clearly see which object to track.\n\n"
            "TASK:\n"
            "1. Look at Image A and understand which object is marked.\n"
            "2. Look at Image B and determine whether the SAME object is visible.\n"
            "3. If the object is visible in Image B, output ONE point coordinate on that object.\n"
            "4. If the object is NOT visible in Image B, answer NOT_VISIBLE.\n\n"
            "OUTPUT FORMAT:\n"
            "- If visible: answer with one coordinate in normalized [0,1] range relative to Image B only, in the form: [(x, y)]\n"
            "- If NOT visible: answer exactly: NOT_VISIBLE\n"
        )
    
    # 测试每个样本
    for i, q in enumerate(tqdm(questions, desc="测试中")):
        scene_id = q.get("scene_id", "")
        uid = q.get("uid", "")
        frame_a_id = q.get("frame_a_id", "")
        frame_b_id = q.get("frame_b_id", "")
        label = q.get("object", "")
        
        # 查找 A 和 B 图
        # A图允许fallback（如果找不到指定frame_id，从five_frames中随机选一个）
        # B图不允许fallback（必须是指定的frame_id）
        image_a_path = find_fiveframes_image(fiveframes_root, scene_id, uid, frame_a_id, allow_fallback=True)
        image_b_path = find_fiveframes_image(fiveframes_root, scene_id, uid, frame_b_id, allow_fallback=False)
        
        if image_a_path is None or image_b_path is None:
            print(f"⚠️  样本 {q['id']}: 找不到图像")
            print(f"   A: {image_a_path}, B: {image_b_path}")
            failed += 1
            results.append({
                "question_id": q["id"],
                "text": "ERROR: Image not found",
                "model_id": "fiveframes_multiimage",
                "rgb_path": q.get("rgb_path", ""),
                "mask_path": q.get("mask_path", ""),
            })
            continue
        
        # 构造 prompt
        prompt = build_multiimage_prompt(label)
        
        # 推理
        try:
            output = inference_multi_image(
                model, tokenizer, image_processor,
                image_a_path, image_b_path, prompt, device
            )
        except Exception as e:
            import traceback
            print(f"❌ 样本 {q['id']} 推理失败: {e}")
            print(f"   详细错误: {traceback.format_exc()}")
            failed += 1
            output = f"ERROR: {str(e)}"
        
        # 提取GT坐标（从five_frames数据中）
        gt_coord = None
        try:
            five_frames_file = fiveframes_root / scene_id / f"uid_{uid}" / f"{scene_id}_uid_{uid}_five_frames.json"
            if five_frames_file.exists():
                with open(five_frames_file, 'r') as f:
                    five_frames_data = json.load(f)
                    # 找到frame_b_id对应的数据
                    for frame_data in five_frames_data:
                        if frame_data.get("frame_id") == frame_b_id:
                            gt_coord = (frame_data.get("cx_norm"), frame_data.get("cy_norm"))
                            break
        except Exception as e:
            pass  # 如果找不到GT坐标，继续处理
        
        # 保存结果（包含GT坐标用于后续评估）
        result = {
            "question_id": q["id"],
            "prompt": prompt,
            "object_name": label,
            "text": output,
            "model_id": "fiveframes_multiimage",
            "rgb_path": q.get("rgb_path", ""),
            "mask_path": q.get("mask_path", ""),
            "category": q.get("category", ""),
            "step": q.get("step", 1),
        }
        
        # 如果有GT坐标，添加到结果中
        if gt_coord:
            result["gt_coord"] = gt_coord
        
        results.append(result)
    
    # 保存结果
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 测试完成!")
    print(f"   - 总样本数: {len(questions)}")
    print(f"   - 成功: {len(questions) - failed}")
    print(f"   - 失败: {failed}")
    print(f"   - 结果保存到: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="使用训练后的 five_frames multi-image 模型测试 CrossView benchmark"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练后的模型路径，例如: runs/train/RoboRefer-2B-FiveFrames-MultiImage/model"
    )
    parser.add_argument(
        "--question_json",
        type=str,
        required=True,
        help="CrossView question.json 路径"
    )
    parser.add_argument(
        "--fiveframes_root",
        type=str,
        required=True,
        help="five_frames 数据根目录"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="输出结果 JSON 路径（JSONL 格式）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大测试样本数（用于快速测试）"
    )
    
    args = parser.parse_args()
    
    test_crossview_benchmark(
        model_path=args.model_path,
        question_json=Path(args.question_json),
        fiveframes_root=Path(args.fiveframes_root),
        output_json=Path(args.output_json),
        device=args.device,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
