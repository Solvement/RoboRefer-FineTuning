import os
import re
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Callable
import cv2


def text2pts(text: str, width=640, height=480, is_absolute=False) -> np.ndarray:
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []

    for match in matches:
        vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
        if len(vector) == 2:
            x, y = vector
            if not is_absolute and (isinstance(x, float) or isinstance(y, float)):
                x = int(x * width)
                y = int(y * height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if not is_absolute:
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            y, x = np.where(np.ones((y1 - y0, x1 - x0)))
            points.extend(list(np.stack([x + x0, y + y0], axis=1)))

    return np.array(points)


def xml2pts(text: str, width: int, height: int) -> np.ndarray:
    pattern = re.compile(r'(x\d+)="(-?\d+\.?\d*)"\s+(y\d+)="(-?\d+\.?\d*)"')
    matches = pattern.findall(text)
    return np.array([
        (int(float(x) / 100 * width), int(float(y) / 100 * height))
        for _, x, _, y in matches
    ])


def json2pts(text: str, width=640, height=480) -> np.ndarray:
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if not match:
        return np.empty((0, 2), dtype=int)
    
    try:
        data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return np.empty((0, 2), dtype=int)

    points = []
    for item in data:
        if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
            y_norm, x_norm = item["point"]
            x = int(x_norm / 1000 * width)
            y = int(y_norm / 1000 * height)
            points.append((x, y))
    return np.array(points)


def right_to_full_xy(x_norm, y_norm, img_a_width_norm=0.5):
    """
    将右半张图（图B）的归一化坐标转换为整张拼接图的归一化坐标
    
    Args:
        x_norm: 右半张图的x坐标（归一化，[0,1]）
        y_norm: 右半张图的y坐标（归一化，[0,1]）
        img_a_width_norm: 左半张图（图A）宽度占整图的比例
    
    Returns:
        x_full, y_full: 整张拼接图的归一化坐标
    """
    # 假设左右等宽拼接，无间隔
    x_full = img_a_width_norm + x_norm * (1 - img_a_width_norm)
    y_full = y_norm
    return x_full, y_full


def compute_accuracy(
    answers: List[Dict],
    task_name: str,
    parse_func: Callable[[str, int, int], np.ndarray],
    question_metadata: Dict = None,
    coord_frame: str = "full"
) -> None:
    """
    计算准确率（多指标：Mask-success + Mask-proximity + 两种距离指标）
    
    Args:
        answers: 模型输出结果列表
        task_name: 任务名称
        parse_func: 解析函数，将文本转换为坐标点
        question_metadata: question.json 的元数据（可选，用于拼接模式）
        coord_frame: 坐标系统，"full"或"right"
    """
    mask_success = []  # 预测点是否在mask内（或负例正确输出NOT_VISIBLE）
    mask_success_dilated = []  # 预测点是否在dilated mask内（诊断用）
    mask_proximities = []  # 预测点到mask最近距离（归一化）
    dist_to_mask_center = []  # 预测点到确定性GT点（mask内点中心）的距离
    dist_to_bbox_center = []  # 预测点到bbox center的距离（参考字段）
    negative_correct = []  # 负例是否正确输出NOT_VISIBLE
    
    # 如果提供了 question_metadata，加载它（用于拼接模式）
    if question_metadata is None:
        question_file = os.path.join("./RefSpatial-Bench", task_name, "question.json")
        if os.path.exists(question_file):
            with open(question_file, 'r') as f:
                question_metadata = {q["id"]: q for q in json.load(f)}

    for answer in tqdm(answers):
        mask_path = os.path.join("./RefSpatial-Bench", task_name, answer['mask_path'])
        mask = np.array(Image.open(mask_path)) / 255.
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)

        try:
            # 尝试解析JSON格式的二阶段输出
            text = answer["text"].strip()
            is_json_format = False
            visible = None
            point = None
            
            # 尝试提取JSON对象（更宽松的匹配）
            # 先尝试完整JSON对象
            json_match = re.search(r'\{[^{}]*"visible"[^{}]*"point"[^{}]*\}', text, re.DOTALL)
            if not json_match:
                # 尝试只有visible的JSON
                json_match = re.search(r'\{[^{}]*"visible"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    if 'visible' in parsed:
                        is_json_format = True
                        visible = parsed.get('visible', False)
                        point = parsed.get('point', None)
                except Exception as e:
                    # 如果JSON解析失败，尝试手动提取
                    pass
            
            # 如果还是没找到，尝试从文本中提取（可能是多行JSON或非标准格式）
            if not is_json_format:
                # 尝试提取完整的JSON块（可能跨多行）
                json_block = re.search(r'\{.*?"visible".*?\}', text, re.DOTALL)
                if json_block:
                    try:
                        json_str = json_block.group(0)
                        parsed = json.loads(json_str)
                        if 'visible' in parsed:
                            is_json_format = True
                            visible = parsed.get('visible', False)
                            point = parsed.get('point', None)
                    except:
                        pass
                
                # 如果还是没找到，尝试解析非标准格式（如 {visible: true, point: [0.5, 0.5]}）
                if not is_json_format:
                    # 提取visible值
                    visible_match = re.search(r'visible\s*:\s*(true|false)', text, re.IGNORECASE)
                    if visible_match:
                        visible_str = visible_match.group(1).lower()
                        visible = visible_str == 'true'
                        is_json_format = True
                        
                        # 提取point值
                        point_match = re.search(r'point\s*:\s*\[?\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]?', text)
                        if point_match:
                            x, y = float(point_match.group(1)), float(point_match.group(2))
                            point = [x, y]
                        else:
                            # 检查是否有null
                            if 'null' in text.lower() or 'point' not in text.lower():
                                point = None
                            else:
                                point = None
            
            # 如果不是JSON格式，回退到旧格式
            if not is_json_format:
                # 检查是否输出NOT_VISIBLE（文本格式）
                text_lower = text.lower()
                is_not_visible = "not_visible" in text_lower or "not visible" in text_lower or "not found" in text_lower
                
                # 解析模型输出的坐标（归一化坐标）
                pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
                matches = re.findall(pattern, text)
                norm_points = []
                for match in matches:
                    vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
                    if len(vector) == 2:
                        x, y = vector
                        if isinstance(x, float) or isinstance(y, float):
                            norm_points.append((x, y))
                
                # 转换为JSON格式的表示
                if is_not_visible:
                    visible = False
                    point = None
                elif norm_points:
                    visible = True
                    point = norm_points[0]
                else:
                    visible = None
                    point = None
            
            # 如果是负例，检查是否正确输出NOT_VISIBLE
            question = question_metadata.get(answer['question_id']) if question_metadata else None
            is_negative = question.get('is_negative', False) if question else False
            expected_not_visible = question.get('expected_answer') == 'NOT_VISIBLE' if question else False
            
            # 处理NOT_VISIBLE情况
            if visible is False or (is_negative and expected_not_visible and visible is not True):
                # 处理NOT_VISIBLE情况
                if is_negative and expected_not_visible:
                    # 负例：正确输出NOT_VISIBLE（visible=false）算成功
                    mask_success_val = 1.0 if visible is False else 0.0
                else:
                    # 正例：输出NOT_VISIBLE算失败
                    mask_success_val = 0.0
                mask_proximity = None
                dist_mask_center = None
                dist_bbox_center = None
                mask_success_dilated_val = None
            elif point is None or visible is False:
                mask_success_val = 0.0
                mask_proximity = None
                dist_mask_center = None
                dist_bbox_center = None
                mask_success_dilated_val = None
            else:
                # 取点坐标（来自JSON或旧格式）
                if point is not None:
                    x_norm, y_norm = point[0], point[1]
                else:
                    # 回退：尝试从文本提取
                    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
                    matches = re.findall(pattern, text)
                    if matches:
                        vector = [float(num) if '.' in num else int(num) for num in matches[0].split(',')]
                        if len(vector) == 2:
                            x_norm, y_norm = vector[0], vector[1]
                        else:
                            raise ValueError("Cannot parse coordinates")
                    else:
                        raise ValueError("No coordinates found")
                
                # 根据coord_frame参数决定坐标解释方式
                if question is None:
                    question = question_metadata.get(answer['question_id']) if question_metadata else None
                img_a_width_norm = question.get('img_a_width_norm', 0.5) if question else 0.5
                
                if coord_frame == "right":
                    # 假设模型输出的是右半张图坐标，转换为整图坐标
                    x_norm, y_norm = right_to_full_xy(x_norm, y_norm, img_a_width_norm)
                # else: coord_frame == "full"，直接使用归一化坐标（已经是整图坐标）
                
                # 转换为像素坐标（相对于整图）
                x_pixel = int(x_norm * mask.shape[1])
                y_pixel = int(y_norm * mask.shape[0])
                
                # 检查点是否在图像范围内
                in_range = (0 <= x_pixel < mask.shape[1]) and (0 <= y_pixel < mask.shape[0])
                
                # 准备mask_binary（用于后续计算）
                mask_binary = (mask > 0).astype(np.uint8)
                
                # 指标1: Mask-success（预测点是否在mask内）
                mask_success_val = float(mask[y_pixel, x_pixel]) if in_range else 0.0
                
                # 指标1b: Mask-success with dilation（诊断用，不改变主指标）
                mask_success_dilated_val = None
                if mask_binary.sum() > 0:
                    # 对mask进行dilation（r=3像素）
                    kernel_size = 7  # 约3像素半径
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
                    mask_success_dilated_val = float(mask_dilated[y_pixel, x_pixel]) if in_range else 0.0
                
                # 指标2: Mask-proximity（预测点到mask最近距离，归一化）
                # 使用distance transform计算到mask最近像素的距离
                # mask_binary已在上面定义
                if mask_binary.sum() > 0:
                    if in_range:
                        # 计算到mask最近像素的距离
                        # 对于mask外的点：使用distance transform（到mask边界的距离）
                        # 对于mask内的点：距离为0（已经在mask内）
                        if mask_binary[y_pixel, x_pixel] > 0:
                            # 预测点在mask内，距离为0
                            mask_proximity = 0.0
                        else:
                            # 预测点在mask外，计算到mask边界的距离
                            dist_transform = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)
                            dist_pixel = dist_transform[y_pixel, x_pixel]
                            # 归一化到[0,1]（除以图像对角线长度）
                            img_diag = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
                            mask_proximity = dist_pixel / img_diag if img_diag > 0 else 1.0
                    else:
                        # 预测点超出图像范围，设为最大距离
                        mask_proximity = 1.0
                else:
                    mask_proximity = None
                
                # 指标3 & 4: 两种距离指标
                dist_mask_center = None
                dist_bbox_center = None
                if question:
                    # 获取确定性GT点（mask内点中心）
                    gt_point_mask_center = None
                    if coord_frame == "right":
                        gt_point_norm = question.get('gt_point_norm')
                        if gt_point_norm:
                            img_a_width_norm = question.get('img_a_width_norm', 0.5)
                            gt_x_full, gt_y_full = right_to_full_xy(gt_point_norm[0], gt_point_norm[1], img_a_width_norm)
                            gt_point_mask_center = (gt_x_full, gt_y_full)
                    else:
                        gt_point_concat = question.get('gt_point_concat')
                        if gt_point_concat:
                            gt_point_mask_center = (gt_point_concat[0], gt_point_concat[1])
                    
                    # 获取bbox center（参考字段）
                    gt_point_bbox_center = None
                    if coord_frame == "right":
                        bbox_center_norm = question.get('gt_point_bbox_center_norm')
                        if bbox_center_norm:
                            img_a_width_norm = question.get('img_a_width_norm', 0.5)
                            bbox_x_full, bbox_y_full = right_to_full_xy(bbox_center_norm[0], bbox_center_norm[1], img_a_width_norm)
                            gt_point_bbox_center = (bbox_x_full, bbox_y_full)
                    else:
                        bbox_center_concat = question.get('gt_point_bbox_center_concat')
                        if bbox_center_concat:
                            gt_point_bbox_center = (bbox_center_concat[0], bbox_center_concat[1])
                    
                    # 计算到mask center的距离
                    if gt_point_mask_center:
                        pred_x_norm, pred_y_norm = x_norm, y_norm
                        gt_x_norm, gt_y_norm = gt_point_mask_center
                        dx = pred_x_norm - gt_x_norm
                        dy = pred_y_norm - gt_y_norm
                        dist_mask_center = np.sqrt(dx*dx + dy*dy) / np.sqrt(2.0)
                    
                    # 计算到bbox center的距离
                    if gt_point_bbox_center:
                        pred_x_norm, pred_y_norm = x_norm, y_norm
                        bbox_x_norm, bbox_y_norm = gt_point_bbox_center
                        dx = pred_x_norm - bbox_x_norm
                        dy = pred_y_norm - bbox_y_norm
                        dist_bbox_center = np.sqrt(dx*dx + dy*dy) / np.sqrt(2.0)
        except Exception as e:
            print(f"Failed to parse question {answer['question_id']}: {e}")
            mask_success_val = 0.0
            mask_proximity = None
            dist_mask_center = None
            dist_bbox_center = None

        mask_success.append(mask_success_val)
        if mask_success_dilated_val is not None:
            mask_success_dilated.append(mask_success_dilated_val)
        if mask_proximity is not None:
            mask_proximities.append(mask_proximity)
        if dist_mask_center is not None:
            dist_to_mask_center.append(dist_mask_center)
        if dist_bbox_center is not None:
            dist_to_bbox_center.append(dist_bbox_center)
        
        # 记录负例正确率
        if is_negative and expected_not_visible:
            negative_correct.append(1.0 if is_not_visible else 0.0)
        
        # 向后兼容：保留accuracy字段
        answer["accuracy"] = mask_success_val
        answer["mask_success"] = mask_success_val
        if mask_proximity is not None:
            answer["mask_proximity"] = mask_proximity
        if dist_mask_center is not None:
            answer["dist_to_mask_center"] = dist_mask_center
        if dist_bbox_center is not None:
            answer["dist_to_bbox_center"] = dist_bbox_center

    # 输出多指标
    print(f"\n{'='*80}")
    print(f"多指标评测结果 (Coord Frame: {coord_frame})")
    print(f"{'='*80}")
    
    # 指标1: Mask-success（主指标）
    print(f"1. Mask-success: {np.mean(mask_success):.4f} ({np.mean(mask_success)*100:.2f}%)")
    print(f"   - 预测点落在mask内的比例（主指标）")
    print(f"   - 对于负例：正确输出NOT_VISIBLE算成功")
    print(f"   - Evaluated: {len(mask_success)}, Total: {len(answers)}")
    
    # 负例正确率
    if negative_correct:
        neg_acc = np.mean(negative_correct)
        print(f"   - 负例正确率: {neg_acc:.4f} ({neg_acc*100:.2f}%) - {len(negative_correct)}个负例")
    
    # 指标1b: Mask-success with dilation（诊断用）
    if mask_success_dilated:
        dilated_acc = np.mean(mask_success_dilated)
        improvement = dilated_acc - np.mean(mask_success)
        print(f"\n1b. Mask-success (dilated, r=3px): {dilated_acc:.4f} ({dilated_acc*100:.2f}%)")
        print(f"   - 诊断指标：如果显著高于原始mask-success，说明模型已接近，只是边界太苛刻")
        print(f"   - 提升: {improvement:.4f} ({improvement*100:.2f}%)")
        print(f"   - Evaluated: {len(mask_success_dilated)}, Total: {len(answers)}")
    
    # 指标2: Mask-proximity（诊断指标）
    if mask_proximities:
        mean_prox = np.mean(mask_proximities)
        print(f"\n2. Mask-proximity: {mean_prox:.4f} (mean normalized distance to mask)")
        print(f"   - 预测点到mask最近距离的平均值（归一化）")
        print(f"   - 越小越好：<0.02表示非常接近mask，<0.05表示较近")
        print(f"   - Evaluated: {len(mask_proximities)}, Total: {len(answers)}")
        
        # 距离分布
        prox_array = np.array(mask_proximities)
        print(f"   - <0.02 (非常近): {(prox_array < 0.02).sum()}/{len(prox_array)} ({(prox_array < 0.02).sum()/len(prox_array)*100:.1f}%)")
        print(f"   - <0.05 (较近): {(prox_array < 0.05).sum()}/{len(prox_array)} ({(prox_array < 0.05).sum()/len(prox_array)*100:.1f}%)")
        print(f"   - <0.1 (中等): {(prox_array < 0.1).sum()}/{len(prox_array)} ({(prox_array < 0.1).sum()/len(prox_array)*100:.1f}%)")
    else:
        print(f"\n2. Mask-proximity: N/A (mask not available)")
    
    # 指标3: Dist to mask center（确定性GT点）
    if dist_to_mask_center:
        mean_dist_mask = np.mean(dist_to_mask_center)
        print(f"\n3. Dist to mask center: {mean_dist_mask:.4f} (mean normalized L2)")
        print(f"   - 预测点到确定性GT点（mask内点中心）的平均归一化L2距离")
        print(f"   - Evaluated: {len(dist_to_mask_center)}, Total: {len(answers)}")
        
        threshold = 0.1
        hit_rate_mask = np.mean([d < threshold for d in dist_to_mask_center])
        print(f"   - Hit rate (dist < {threshold}): {hit_rate_mask:.4f} ({hit_rate_mask*100:.2f}%)")
    else:
        print(f"\n3. Dist to mask center: N/A (GT point not available)")
    
    # 指标4: Dist to bbox center（参考字段）
    if dist_to_bbox_center:
        mean_dist_bbox = np.mean(dist_to_bbox_center)
        print(f"\n4. Dist to bbox center: {mean_dist_bbox:.4f} (mean normalized L2)")
        print(f"   - 预测点到bbox center的平均归一化L2距离（参考字段）")
        print(f"   - Evaluated: {len(dist_to_bbox_center)}, Total: {len(answers)}")
        
        threshold = 0.1
        hit_rate_bbox = np.mean([d < threshold for d in dist_to_bbox_center])
        print(f"   - Hit rate (dist < {threshold}): {hit_rate_bbox:.4f} ({hit_rate_bbox*100:.2f}%)")
    else:
        print(f"\n4. Dist to bbox center: N/A (bbox center not available)")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--coord_frame", type=str, default="full",
                        choices=["full", "right"],
                        help="Interpret model output coords as full-image ('full') or right-half ('right') coords.")
    args = parser.parse_args()

    answer_file = os.path.join('./outputs', f"{args.model_name}/{args.task_name}.jsonl")
    with open(answer_file, 'r') as f:
        answers = [json.loads(line) for line in f]

    model = args.model_name

    if any(key in model for key in ["RoboPoint", "Claude", "GPT4O", "RoboRefer"]):
        compute_accuracy(answers, args.task_name, lambda text, w, h: text2pts(text, w, h, is_absolute=False), coord_frame=args.coord_frame)
    elif any(key in model for key in ["RoboBrain", "Qwen"]):
        compute_accuracy(answers, args.task_name, lambda text, w, h: text2pts(text, w, h, is_absolute=True), coord_frame=args.coord_frame)
    elif "Molmo" in model:
        compute_accuracy(answers, args.task_name, xml2pts, coord_frame=args.coord_frame)
    elif "Gemini" in model:
        compute_accuracy(answers, args.task_name, json2pts, coord_frame=args.coord_frame)
    else:
        print(f"Unknown model type: {model}")


if __name__ == '__main__':
    main()