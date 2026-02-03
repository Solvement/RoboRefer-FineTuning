#!/usr/bin/env python3
"""
多视角一致性评估脚本

独立验证脚本，不依赖训练流程，输出指标和每样本预测结果。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_image
from llava.model import LlavaLlamaModel


NOT_VISIBLE_TOKEN = "NOT_VISIBLE"
COORD_PATTERN = re.compile(r"\[\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\]")


def parse_prediction(text: str) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """
    解析模型预测
    返回: (is_not_visible, coord_tuple) 或 (False, None) 如果无法解析
    """
    text_upper = text.upper()
    if NOT_VISIBLE_TOKEN in text_upper:
        return True, None
    
    # 尝试解析坐标
    match = COORD_PATTERN.search(text)
    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            return False, (x, y)
        except ValueError:
            pass
    
    return False, None


def parse_gt(gt_text: str) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """解析GT标签"""
    if gt_text.upper() == NOT_VISIBLE_TOKEN:
        return True, None
    
    match = COORD_PATTERN.search(gt_text)
    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            return False, (x, y)
        except ValueError:
            pass
    
    return False, None


def compute_l2_error(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """计算L2误差"""
    return np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """加载模型和tokenizer"""
    from transformers import AutoConfig
    
    model_path = Path(model_path)
    
    # 检查是否有model子目录（训练输出格式）
    if (model_path / "model").exists():
        model_path = model_path / "model"
    
    # 加载config
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    config.resume_path = str(model_path)
    
    # 加载模型
    model = LlavaLlamaModel.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    # 获取tokenizer（模型初始化时已加载）
    tokenizer = model.tokenizer
    
    # 设置chat template（如果未设置）
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        chat_template_path = Path(__file__).parent.parent / "llava" / "model" / "language_model" / "chat_templates" / "qwen2.jinja"
        if chat_template_path.exists():
            with open(chat_template_path) as f:
                chat_template = f.read().replace("    ", "").replace("\n", "")
            tokenizer.chat_template = chat_template
    
    return model, tokenizer


def build_prompt_with_images(human_prompt: str, num_images: int = 2) -> str:
    """
    构建包含<image> token的prompt
    注意：dataset的preprocess_rgbd会在文本前插入<image>\n，这里需要模拟相同行为
    """
    # 在prompt前插入num_images个<image>\n
    image_tokens = f"{DEFAULT_IMAGE_TOKEN}\n" * num_images
    return image_tokens + human_prompt


def format_conversation(prompt: str, tokenizer) -> str:
    """使用chat template格式化对话"""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": None}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def run_inference(
    model: LlavaLlamaModel,
    tokenizer: AutoTokenizer,
    sample: Dict,
    image_folder: str,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> str:
    """运行推理"""
    model.eval()
    with torch.no_grad():
        # 获取human prompt
        human_prompt = sample["conversations"][0]["value"]
        
        # 构建包含<image> token的prompt（2个图像）
        prompt_with_images = build_prompt_with_images(human_prompt, num_images=2)
        
        # 使用chat template格式化
        prompt_str = format_conversation(prompt_with_images, tokenizer)
        
        # 加载图像
        image_paths = sample.get("image", [])
        if len(image_paths) != 2:
            raise ValueError(f"Expected 2 images, got {len(image_paths)}")
        
        images = []
        for img_path in image_paths:
            # 使用绝对路径，image_folder设为"/"（会被忽略）
            img = process_image(img_path, None, image_folder)
            images.append(img)
        
        images_tensor = torch.stack(images).unsqueeze(0).to(device)
        
        # Tokenize
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        
        # Generate
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        
        out = model.generate(
            **inputs,
            images=images_tensor,
            depths=None,  # 不使用depth
            **gen_kwargs,
        )
        
        # Decode
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        
        # 移除prompt部分
        if prompt_str in text:
            text = text.split(prompt_str, 1)[-1]
        
        return text.strip()


def main():
    parser = argparse.ArgumentParser(description="多视角一致性评估")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--data_path", type=str, required=True,
                       help="验证数据JSON路径")
    parser.add_argument("--image_folder", type=str, default="/",
                       help="图像文件夹（实际使用绝对路径，此参数保留兼容性）")
    parser.add_argument("--chat_template", type=str, default="qwen2",
                       help="Chat template名称")
    parser.add_argument("--image_aspect_ratio", type=str, default="dynamic",
                       help="图像宽高比处理方式")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="最大生成token数")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制评估样本数（用于快速测试）")
    parser.add_argument("--output_json", type=str, required=True,
                       help="输出预测结果JSON路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    
    args = parser.parse_args()
    
    print("="*70)
    print("多视角一致性评估")
    print("="*70)
    print(f"模型路径: {args.model_name_or_path}")
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_json}")
    print()
    
    # 加载数据
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    if args.limit:
        data = data[:args.limit]
        print(f"⚠️  限制评估样本数: {args.limit}")
    
    print(f"总样本数: {len(data)}")
    print()
    
    # 加载模型
    print("加载模型...")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.device)
    print("✅ 模型加载完成")
    print()
    
    # 评估
    results = []
    stats = {
        "total": 0,
        "not_visible_gt": 0,
        "not_visible_pred": 0,
        "not_visible_correct": 0,
        "visible_gt": 0,
        "visible_pred": 0,
        "l2_errors": [],
        "success_002": 0,
        "success_005": 0,
        "success_010": 0,
    }
    
    print("开始评估...")
    for i, sample in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i+1}/{len(data)}")
        
        sample_id = sample.get("id", f"sample_{i}")
        gt_text = sample["conversations"][1]["value"]
        
        # 解析GT
        gt_is_nv, gt_coord = parse_gt(gt_text)
        
        # 运行推理
        try:
            pred_text = run_inference(
                model, tokenizer, sample, args.image_folder,
                device=args.device, max_new_tokens=args.max_new_tokens
            )
        except Exception as e:
            print(f"⚠️  样本 {sample_id} 推理失败: {e}")
            pred_text = ""
            pred_is_nv = False
            pred_coord = None
        else:
            # 解析预测
            pred_is_nv, pred_coord = parse_prediction(pred_text)
        
        # 计算指标
        stats["total"] += 1
        
        result = {
            "id": sample_id,
            "pred_text": pred_text,
            "pred_parsed": {
                "is_not_visible": pred_is_nv,
                "coord": pred_coord
            },
            "gt": {
                "text": gt_text,
                "is_not_visible": gt_is_nv,
                "coord": gt_coord
            },
            "success": False,
            "l2_error": None,
        }
        
        if gt_is_nv:
            stats["not_visible_gt"] += 1
            if pred_is_nv:
                stats["not_visible_correct"] += 1
                result["success"] = True
        else:
            stats["visible_gt"] += 1
            if not pred_is_nv and pred_coord is not None:
                # 计算L2误差
                l2 = compute_l2_error(pred_coord, gt_coord)
                stats["l2_errors"].append(l2)
                result["l2_error"] = float(l2)
                
                # Success@threshold
                if l2 <= 0.02:
                    stats["success_002"] += 1
                if l2 <= 0.05:
                    stats["success_005"] += 1
                if l2 <= 0.10:
                    stats["success_010"] += 1
                    result["success"] = True
        
        if pred_is_nv:
            stats["not_visible_pred"] += 1
        
        results.append(result)
    
    print()
    print("="*70)
    print("评估结果")
    print("="*70)
    
    # 计算指标
    not_visible_acc = stats["not_visible_correct"] / stats["not_visible_gt"] if stats["not_visible_gt"] > 0 else 0.0
    mean_l2 = np.mean(stats["l2_errors"]) if stats["l2_errors"] else None
    std_l2 = np.std(stats["l2_errors"]) if stats["l2_errors"] else None
    success_002 = stats["success_002"] / stats["visible_gt"] if stats["visible_gt"] > 0 else 0.0
    success_005 = stats["success_005"] / stats["visible_gt"] if stats["visible_gt"] > 0 else 0.0
    success_010 = stats["success_010"] / stats["visible_gt"] if stats["visible_gt"] > 0 else 0.0
    
    print(f"总样本数: {stats['total']}")
    print(f"NOT_VISIBLE GT: {stats['not_visible_gt']}")
    print(f"可见样本 GT: {stats['visible_gt']}")
    print()
    print(f"NOT_VISIBLE 准确率: {not_visible_acc:.4f} ({stats['not_visible_correct']}/{stats['not_visible_gt']})")
    print()
    if mean_l2 is not None:
        print(f"可见样本点误差 (L2):")
        print(f"  均值: {mean_l2:.4f}")
        print(f"  标准差: {std_l2:.4f}")
        print(f"  中位数: {np.median(stats['l2_errors']):.4f}")
        print()
    print(f"Success@0.02: {success_002:.4f} ({stats['success_002']}/{stats['visible_gt']})")
    print(f"Success@0.05: {success_005:.4f} ({stats['success_005']}/{stats['visible_gt']})")
    print(f"Success@0.10: {success_010:.4f} ({stats['success_010']}/{stats['visible_gt']})")
    print("="*70)
    
    # 保存结果
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metrics": {
            "not_visible_accuracy": float(not_visible_acc),
            "mean_l2_error": float(mean_l2) if mean_l2 is not None else None,
            "std_l2_error": float(std_l2) if std_l2 is not None else None,
            "success_002": float(success_002),
            "success_005": float(success_005),
            "success_010": float(success_010),
        },
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
