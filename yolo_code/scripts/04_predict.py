import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from ultralytics.models import YOLO
import cv2
import argparse
import logging
import matplotlib
import platform
import traceback
from datetime import datetime
import random
import json
from collections import Counter
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
import pandas as pd
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

def setup_matplotlib_fonts(font_path=None):
    """设置matplotlib字体，兼容不同操作系统"""
    system = platform.system()
    
    try:
        if font_path and Path(font_path).exists():
            # 如果指定了字体文件并且存在，优先使用它
            matplotlib.font_manager.fontManager.addfont(str(font_path))
            plt.rcParams['font.sans-serif'] = [Path(font_path).stem]
        elif system == 'Windows':
            # Windows默认字体路径
            win_font_path = 'C:/Windows/Fonts/simhei.ttf'
            if Path(win_font_path).exists():
                matplotlib.font_manager.fontManager.addfont(win_font_path)
                plt.rcParams['font.sans-serif'] = ['SimHei']
            else:
                plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
        elif system == 'Linux':
            # Linux常见中文字体
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'SimHei']
        else:  # macOS或其他系统
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            
        plt.rcParams['axes.unicode_minus'] = False
        # 使用更美观的样式
        plt.style.use('ggplot')
        return True
    except Exception as e:
        print(f"字体设置失败: {e}, 可能会导致中文显示异常")
        return False

def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def read_yolo_label(label_path, img_width, img_height, class_names):
    """读取YOLO格式的标签文件，返回标签数据"""
    annotations = []
    
    if not Path(label_path).exists():
        return annotations
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id < len(class_names):  # 确保类别ID有效
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        bbox_width = float(parts[3]) * img_width
                        bbox_height = float(parts[4]) * img_height
                        
                        # 计算坐标
                        xmin = max(0, x_center - bbox_width / 2)
                        ymin = max(0, y_center - bbox_height / 2)
                        xmax = min(img_width, x_center + bbox_width / 2)
                        ymax = min(img_height, y_center + bbox_height / 2)
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_names[class_id],
                            'bbox': [xmin, ymin, xmax, ymax],
                            'confidence': 1.0  # 真实标签置信度为1
                        })
    except Exception as e:
        print(f"读取标签文件 {label_path} 时出错: {e}")
    
    return annotations

def calculate_metrics(true_annotations, pred_annotations, iou_threshold=0.5):
    """
    计算预测结果与真实标签的差异度量
    
    参数:
    - true_annotations: 真实标注列表
    - pred_annotations: 预测标注列表
    - iou_threshold: IoU阈值，用于判断检测是否正确
    
    返回:
    - metrics: 包含精度、召回率、F1分数和类别统计的字典
    """
    metrics = {
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'class_stats': {},
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'iou_scores': [],      # 存储每个匹配对的IoU得分
        'confidence_scores': [] # 存储每个预测的置信度
    }
    
    # 如果没有真实标注和预测标注
    if not true_annotations and not pred_annotations:
        metrics['precision'] = 1.0
        metrics['recall'] = 1.0
        metrics['f1_score'] = 1.0
        return metrics
    
    # 如果只有真实标注，没有预测标注
    if true_annotations and not pred_annotations:
        metrics['precision'] = 0
        metrics['recall'] = 0
        metrics['f1_score'] = 0
        metrics['false_negatives'] = len(true_annotations)
        return metrics
    
    # 如果只有预测标注，没有真实标注
    if not true_annotations and pred_annotations:
        metrics['precision'] = 0
        metrics['recall'] = 0
        metrics['f1_score'] = 0
        metrics['false_positives'] = len(pred_annotations)
        return metrics
    
    # 存储预测标注的置信度
    for pred in pred_annotations:
        metrics['confidence_scores'].append(pred['confidence'])
    
    # 创建标记列表，用于跟踪哪些真实标注已被匹配
    matched_true = [False] * len(true_annotations)
    true_positives = 0
    false_positives = 0
    
    # 类别统计
    class_stats = {}
    true_classes = Counter([anno['class_name'] for anno in true_annotations])
    pred_classes = Counter([anno['class_name'] for anno in pred_annotations])
    
    for class_name in set(list(true_classes.keys()) + list(pred_classes.keys())):
        class_stats[class_name] = {
            'true': true_classes.get(class_name, 0),
            'predicted': pred_classes.get(class_name, 0),
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'iou_scores': []  # 存储每个类别的IoU得分
        }
    
    # 对每个预测标注
    for pred_idx, pred in enumerate(pred_annotations):
        matched = False
        best_iou = 0
        best_idx = -1
        
        for true_idx, true in enumerate(true_annotations):
            # 计算IoU
            if pred['class_id'] == true['class_id'] and not matched_true[true_idx]:
                # 计算交集面积
                x1 = max(pred['bbox'][0], true['bbox'][0])
                y1 = max(pred['bbox'][1], true['bbox'][1])
                x2 = min(pred['bbox'][2], true['bbox'][2])
                y2 = min(pred['bbox'][3], true['bbox'][3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    
                    # 计算两个框的面积
                    pred_area = (pred['bbox'][2] - pred['bbox'][0]) * (pred['bbox'][3] - pred['bbox'][1])
                    true_area = (true['bbox'][2] - true['bbox'][0]) * (true['bbox'][3] - true['bbox'][1])
                    
                    # 计算并集面积
                    union = pred_area + true_area - intersection
                    
                    # 计算IoU
                    iou = intersection / union
                    
                    if iou > iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = true_idx
                        matched = True
        
        if matched:
            # 匹配成功
            matched_true[best_idx] = True
            true_positives += 1
            class_name = pred['class_name']
            class_stats[class_name]['true_positives'] += 1
            # 记录IoU得分
            metrics['iou_scores'].append(best_iou)
            class_stats[class_name]['iou_scores'].append(best_iou)
        else:
            # 假阳性
            false_positives += 1
            class_name = pred['class_name']
            class_stats[class_name]['false_positives'] += 1
    
    # 计算假阴性
    false_negatives = matched_true.count(False)
    
    # 更新假阴性的类别统计
    for idx, matched in enumerate(matched_true):
        if not matched:
            class_name = true_annotations[idx]['class_name']
            class_stats[class_name]['false_negatives'] += 1
    
    # 计算精确率和召回率
    if true_positives + false_positives > 0:
        metrics['precision'] = true_positives / (true_positives + false_positives)
    else:
        metrics['precision'] = 0
    
    if true_positives + false_negatives > 0:
        metrics['recall'] = true_positives / (true_positives + false_negatives)
    else:
        metrics['recall'] = 0
    
    # 计算F1分数
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
    
    # 更新类别统计
    for class_name, stats in class_stats.items():
        if stats['true_positives'] + stats['false_positives'] > 0:
            stats['precision'] = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
        else:
            stats['precision'] = 0
        
        if stats['true_positives'] + stats['false_negatives'] > 0:
            stats['recall'] = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
        else:
            stats['recall'] = 0
        
        if stats['precision'] + stats['recall'] > 0:
            stats['f1_score'] = 2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall'])
        else:
            stats['f1_score'] = 0
    
    metrics['class_stats'] = class_stats
    metrics['true_positives'] = true_positives
    metrics['false_positives'] = false_positives
    metrics['false_negatives'] = false_negatives
    
    return metrics

def predict_with_ground_truth(model, img_path, label_path, class_names, output_dir, conf_threshold=0.25):
    """预测单张图像并与真实标签对比"""
    try:
        img_path = Path(img_path)
        label_path = Path(label_path)
        img_name = img_path.name
        base_name = img_path.stem
        
        # 加载图像
        img = Image.open(img_path)
        width, height = img.size
        
        # 获取真实标签
        true_annotations = read_yolo_label(label_path, width, height, class_names)
        
        # 进行预测
        results = model(img_path, conf=conf_threshold)
        
        # 提取预测标注
        pred_annotations = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id < len(class_names):
                    pred_annotations.append({
                        'class_id': cls_id,
                        'class_name': class_names[cls_id],
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        # 计算指标
        metrics = calculate_metrics(true_annotations, pred_annotations)
        
        # 获取颜色 - 不再转换为0-255范围，让matplotlib自己处理
        colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
        
        # 绘制对比可视化
        plt.figure(figsize=(15, 8))
        
        # 绘制原始图像和真实标签
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(img))
        plt.title('真实标签')
        plt.axis('off')
        
        # 绘制真实标签边界框
        for anno in true_annotations:
            xmin, ymin, xmax, ymax = anno['bbox']
            cls_id = anno['class_id']
            color = colors[cls_id % len(colors)]
            
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                linewidth=2, edgecolor=color,
                                facecolor='none')
            plt.gca().add_patch(rect)
            
            plt.text(xmin, ymin - 5, class_names[cls_id], 
                     bbox=dict(facecolor=color, alpha=0.5),
                     fontsize=8, color='white')
        
        # 绘制预测结果
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(img))
        plt.title('预测结果')
        plt.axis('off')
        
        # 绘制预测边界框
        for anno in pred_annotations:
            xmin, ymin, xmax, ymax = anno['bbox']
            cls_id = anno['class_id']
            conf = anno['confidence']
            color = colors[cls_id % len(colors)]
            
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                linewidth=2, edgecolor=color,
                                facecolor='none')
            plt.gca().add_patch(rect)
            
            plt.text(xmin, ymin - 5, f'{class_names[cls_id]} {conf:.2f}', 
                     bbox=dict(facecolor=color, alpha=0.5),
                     fontsize=8, color='white')
        
        plt.tight_layout()
        output_path = Path(output_dir) / f"compare_{base_name}.jpg"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # 绘制热力图格式的对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # 显示原始图像
        ax1.imshow(np.array(img))
        ax1.set_title('真实标签热力图')
        ax1.axis('off')
        
        # 创建真实标签热力图
        heatmap = np.zeros((height, width))
        for anno in true_annotations:
            xmin, ymin, xmax, ymax = map(int, anno['bbox'])
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(width, xmax), min(height, ymax)
            heatmap[ymin:ymax, xmin:xmax] += 1
            
        # 使用Alpha混合方式显示热力图
        if heatmap.max() > 0:  # 避免除零错误
            heatmap = heatmap / heatmap.max()
        mask = heatmap > 0
        heatmap_img = np.array(img).copy()
        
        # 将热力图颜色应用到原图上
        heatmap_colored = plt.cm.jet(heatmap)
        heatmap_img[mask] = heatmap_colored[mask, :3] * 255 * 0.7 + heatmap_img[mask] * 0.3
        
        ax1.imshow(heatmap_img.astype(np.uint8))
        
        # 对预测标签做同样处理
        ax2.imshow(np.array(img))
        ax2.set_title('预测结果热力图')
        ax2.axis('off')
        
        # 创建预测标签热力图，使用置信度作为权重
        heatmap = np.zeros((height, width))
        for anno in pred_annotations:
            xmin, ymin, xmax, ymax = map(int, anno['bbox'])
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(width, xmax), min(height, ymax)
            heatmap[ymin:ymax, xmin:xmax] += anno['confidence']
            
        # 使用Alpha混合方式显示热力图
        if heatmap.max() > 0:  # 避免除零错误
            heatmap = heatmap / heatmap.max()
        mask = heatmap > 0
        heatmap_img = np.array(img).copy()
        
        # 将热力图颜色应用到原图上
        heatmap_colored = plt.cm.jet(heatmap)
        heatmap_img[mask] = heatmap_colored[mask, :3] * 255 * 0.7 + heatmap_img[mask] * 0.3
        
        ax2.imshow(heatmap_img.astype(np.uint8))
        
        plt.tight_layout()
        output_heatmap_path = Path(output_dir) / f"heatmap_{base_name}.jpg"
        plt.savefig(output_heatmap_path, dpi=300)
        plt.close()
        
        # 统计检测到的目标
        true_counts = {}
        for anno in true_annotations:
            class_name = anno['class_name']
            if class_name not in true_counts:
                true_counts[class_name] = 0
            true_counts[class_name] += 1
        
        pred_counts = {}
        for anno in pred_annotations:
            class_name = anno['class_name']
            if class_name not in pred_counts:
                pred_counts[class_name] = 0
            pred_counts[class_name] += 1
        
        # 绘制目标统计对比图
        all_classes = sorted(set(list(true_counts.keys()) + list(pred_counts.keys())))
        if all_classes:
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(all_classes))
            width = 0.35
            
            true_values = [true_counts.get(cls, 0) for cls in all_classes]
            pred_values = [pred_counts.get(cls, 0) for cls in all_classes]
            
            plt.bar(x - width/2, true_values, width, label='真实标签', color='tab:blue')
            plt.bar(x + width/2, pred_values, width, label='预测结果', color='tab:orange')
            
            plt.title(f'图像中目标数量对比')
            plt.xlabel('目标类别')
            plt.ylabel('数量')
            plt.xticks(x, all_classes, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            stats_output_path = Path(output_dir) / f"stats_compare_{base_name}.png"
            plt.savefig(stats_output_path, dpi=300)
            plt.close()
            
            # 创建水平条形图比较
            plt.figure(figsize=(12, max(6, len(all_classes) * 0.5)))
            
            y = np.arange(len(all_classes))
            
            plt.barh(y - width/2, true_values, width, label='真实标签', color='tab:blue')
            plt.barh(y + width/2, pred_values, width, label='预测结果', color='tab:orange')
            
            plt.title(f'图像中目标数量对比 (水平)')
            plt.ylabel('目标类别')
            plt.xlabel('数量')
            plt.yticks(y, all_classes)
            plt.legend()
            plt.tight_layout()
            horiz_stats_output_path = Path(output_dir) / f"horiz_stats_compare_{base_name}.png"
            plt.savefig(horiz_stats_output_path, dpi=300)
            plt.close()
        
        # 绘制置信度分布图
        if pred_annotations:
            plt.figure(figsize=(10, 6))
            
            # 按类别分组绘制置信度直方图
            confidence_by_class = {}
            for anno in pred_annotations:
                class_name = anno['class_name']
                if class_name not in confidence_by_class:
                    confidence_by_class[class_name] = []
                confidence_by_class[class_name].append(anno['confidence'])
            
            for i, (class_name, confidences) in enumerate(confidence_by_class.items()):
                plt.hist(confidences, alpha=0.5, bins=10, 
                         label=f'{class_name} (均值={np.mean(confidences):.2f})', 
                         color=colors[i % len(colors)])
            
            plt.title('预测置信度分布')
            plt.xlabel('置信度')
            plt.ylabel('计数')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            conf_output_path = Path(output_dir) / f"confidence_dist_{base_name}.png"
            plt.savefig(conf_output_path, dpi=300)
            plt.close()
            
        # 如果有IoU分数，绘制IoU分布图
        if 'iou_scores' in metrics and metrics['iou_scores']:
            plt.figure(figsize=(10, 6))
            
            plt.hist(metrics['iou_scores'], bins=10, alpha=0.7, color='tab:blue')
            # 使用方法2，将单引号改为双引号
            plt.axvline(x=np.mean(metrics['iou_scores']), color='red', linestyle='--', 
                        label=f'平均IoU: {np.mean(metrics["iou_scores"]):.2f}')
            
            plt.title('检测IoU分布')
            plt.xlabel('IoU值')
            plt.ylabel('频率')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            iou_output_path = Path(output_dir) / f"iou_dist_{base_name}.png"
            plt.savefig(iou_output_path, dpi=300)
            plt.close()
        
        # 创建带标签和预测框的合并图像
        img_with_boxes = np.array(img).copy()
        draw = ImageDraw.Draw(Image.fromarray(img_with_boxes))
        
        # 绘制真实标签边界框 (绿色)
        for anno in true_annotations:
            xmin, ymin, xmax, ymax = map(int, anno['bbox'])
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=3)
            draw.text((xmin, ymin-15), class_names[anno['class_id']], fill=(0, 255, 0))
        
        # 绘制预测边界框 (红色)
        for anno in pred_annotations:
            xmin, ymin, xmax, ymax = map(int, anno['bbox'])
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(255, 0, 0), width=2)
            draw.text((xmin, ymin-15), f"{class_names[anno['class_id']]} {anno['confidence']:.2f}", fill=(255, 0, 0))
        
        # 保存合并图像
        merged_img = Image.fromarray(np.array(draw.im))
        merged_output_path = Path(output_dir) / f"merged_{base_name}.jpg"
        merged_img.save(merged_output_path)
        
        return metrics, true_counts, pred_counts
    
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {e}")
        traceback.print_exc()
        return None, {}, {}

def find_latest_model(base_dir):
    """查找所有可能的模型路径，返回最新的一个"""
    # 模型可能的位置
    base_dir = Path(base_dir)
    possible_locations = [
        # 模型目录中的模型
        list(base_dir.glob('**/models/*.pt')),
        # train目录中的weights目录
        list(base_dir.glob('**/train*/weights/*.pt')),
    ]
    
    all_models = []
    for location in possible_locations:
        all_models.extend(location)
    
    if not all_models:
        return None
    
    # 按修改时间排序，返回最新的
    return sorted(all_models, key=os.path.getmtime)[-1]

def find_latest_train_dir(results_dir):
    """查找数字最大的train目录"""
    train_dirs = []
    for d in results_dir.glob("train*"):
        if d.is_dir():
            # 提取目录名中的数字
            try:
                # 提取目录名中的数字部分 (如 'train11' -> 11)
                num = int(d.name.replace('train', '')) if d.name != 'train' else 0
                train_dirs.append((num, d))
            except ValueError:
                # 如果目录名不符合'trainX'格式，则忽略
                continue
    
    # 按数字大小排序，取最大的
    if train_dirs:
        train_dirs.sort(key=lambda x: x[0], reverse=True)
        print(f"找到的训练目录: {[d[1].name for d in train_dirs]}, 选择: {train_dirs[0][1].name}")
        return train_dirs[0][1]  # 返回目录路径
    return None

def plot_advanced_visualizations(all_metrics, all_true_counts, all_pred_counts, merged_class_stats, class_names, output_dir):
    """生成更多高级可视化图表"""
    # 1. 绘制混淆矩阵
    if merged_class_stats:
        # 提取类别名称列表
        classes = sorted(merged_class_stats.keys())
        
        # 创建混淆矩阵数据
        cm = np.zeros((len(classes), len(classes)))
        
        # 我们把每个类别的真实正例放在对角线上
        # 假阳性按行分布，假阴性按列分布
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                if i == j:  # 对角线元素是真阳性
                    cm[i, j] = merged_class_stats[true_class]['true_positives']
                else:
                    # 这里的计算是一个近似，混淆矩阵需要实例级别的标注关系
                    # 对于目标检测，严格来说我们需要知道错误的类别预测
                    cm[i, j] = 0  # 默认设为0
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title('类别检测混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(str(output_dir / "confusion_matrix.png"), dpi=300)
        plt.close()
        
        # 2. 绘制精确率-召回率曲线 (近似版本，基于类别)
        plt.figure(figsize=(10, 8))
        
        # 使用每个类的精确率和召回率画点
        for class_name, stats in merged_class_stats.items():
            plt.scatter(stats.get('recall', 0), stats.get('precision', 0), 
                       label=class_name, s=100)
        
        # 绘制平均精确率-召回率点
        precisions = [stats.get('precision', 0) for stats in merged_class_stats.values()]
        recalls = [stats.get('recall', 0) for stats in merged_class_stats.values()]
        mean_precision = np.mean(precisions) if precisions else 0
        mean_recall = np.mean(recalls) if recalls else 0
        
        plt.scatter(mean_recall, mean_precision, marker='*', color='red', s=300, label='平均')
        
        plt.title('类别精确率-召回率点图')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(str(output_dir / "pr_scatter.png"), dpi=300)
        plt.close()
        
        # 3. 绘制气泡图，显示每个类的样本数量和性能
        plt.figure(figsize=(12, 8))
        
        x_values = []  # 召回率
        y_values = []  # 精确率
        sizes = []     # 气泡大小 (样本数量)
        colors = []    # 气泡颜色 (F1分数)
        labels = []    # 类别标签
        
        for class_name, stats in merged_class_stats.items():
            if stats['true'] > 0:  # 只关注有真实样本的类别
                x_values.append(stats.get('recall', 0))
                y_values.append(stats.get('precision', 0))
                sizes.append(stats['true'] * 30)  # 乘以一个因子使气泡更明显
                colors.append(stats.get('f1_score', 0))
                labels.append(class_name)
        
        scatter = plt.scatter(x_values, y_values, s=sizes, c=colors, 
                              alpha=0.6, cmap='viridis', edgecolors='black')
        
        # 添加类别标签
        for i, label in enumerate(labels):
            plt.annotate(label, (x_values[i], y_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, label='F1分数')
        plt.title('类别性能气泡图')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(str(output_dir / "performance_bubble.png"), dpi=300)
        plt.close()
        
        # 4. 绘制雷达图，显示每个类别的综合性能
        # 准备雷达图数据
        # 我们将使用精确率、召回率、F1分数作为三个维度
        categories = ['精确率', '召回率', 'F1分数']
        N = len(categories)
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图
        
        # 为每个类别绘制一条雷达线
        for i, (class_name, stats) in enumerate(merged_class_stats.items()):
            if i >= 10:  # 限制类别数量，避免图表过于拥挤
                break
                
            values = [
                stats.get('precision', 0),
                stats.get('recall', 0),
                stats.get('f1_score', 0)
            ]
            values += values[:1]  # 闭合雷达图
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=class_name)
            ax.fill(angles, values, alpha=0.1)
        
        # 设置雷达图的刻度和标签
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
        plt.ylim(0, 1)
        
        plt.title('类别性能雷达图')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(str(output_dir / "radar_chart.png"), dpi=300)
        plt.close()
        
        # 5. 绘制堆叠条形图，显示每个类别的TP、FP、FN
        data = []
        class_labels = []
        
        for class_name, stats in merged_class_stats.items():
            data.append([
                stats['true_positives'],
                stats['false_positives'],
                stats['false_negatives']
            ])
            class_labels.append(class_name)
        
        # 转换为NumPy数组，以便进行堆叠
        data = np.array(data)
        
        # 创建堆叠条形图
        plt.figure(figsize=(12, 8))
        
        # 仅绘制数据非空的类别
        non_empty_indices = [i for i, row in enumerate(data) if any(row)]
        
        if non_empty_indices:
            filtered_data = data[non_empty_indices]
            filtered_labels = [class_labels[i] for i in non_empty_indices]
            
            ind = np.arange(len(filtered_labels))
            width = 0.5
            
            p1 = plt.bar(ind, filtered_data[:, 0], width, label='真阳性', color='tab:green')
            p2 = plt.bar(ind, filtered_data[:, 1], width, bottom=filtered_data[:, 0], 
                          label='假阳性', color='tab:red')
            p3 = plt.bar(ind, filtered_data[:, 2], width, 
                          bottom=filtered_data[:, 0] + filtered_data[:, 1], 
                          label='假阴性', color='tab:blue')
            
            plt.ylabel('样本数量')
            plt.title('各类别检测结果分布')
            plt.xticks(ind, filtered_labels, rotation=45, ha='right')
            plt.legend()
            
            # 在柱状图顶部添加标签
            for i, rect in enumerate(p1.patches):
                height = rect.get_height()
                if height > 0:
                    plt.text(rect.get_x() + rect.get_width() / 2., height / 2,
                            f'{int(height)}', ha='center', va='center', color='white')
            
            for i, rect in enumerate(p2.patches):
                height = rect.get_height()
                bottom = filtered_data[i, 0]
                if height > 0:
                    plt.text(rect.get_x() + rect.get_width() / 2., bottom + height / 2,
                            f'{int(height)}', ha='center', va='center', color='white')
                            
            for i, rect in enumerate(p3.patches):
                height = rect.get_height()
                bottom = filtered_data[i, 0] + filtered_data[i, 1]
                if height > 0:
                    plt.text(rect.get_x() + rect.get_width() / 2., bottom + height / 2,
                            f'{int(height)}', ha='center', va='center', color='white')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "detection_distribution.png"), dpi=300)
            plt.close()
    
    # 6. 绘制置信度与IoU关系散点图
    all_confidences = []
    all_ious = []
    class_ids = []
    
    for metrics in all_metrics:
        if 'iou_scores' in metrics and 'confidence_scores' in metrics:
            # 需要将IoU得分与对应的置信度匹配
            # 但由于我们没有直接的匹配关系，这里只能选择有IoU得分的预测
            # 这意味着这些是真阳性预测
            ious = metrics['iou_scores']
            # 从所有预测中选出置信度最高的几个作为匹配
            # 这里是近似，实际应该有对应关系
            confidences = sorted(metrics['confidence_scores'], reverse=True)[:len(ious)]
            
            if len(confidences) == len(ious):
                all_confidences.extend(confidences)
                all_ious.extend(ious)
                # 这里没有类别ID信息，所以设为0
                class_ids.extend([0] * len(ious))
    
    if all_confidences and all_ious:
        plt.figure(figsize=(10, 8))
        
        # 为不同置信度区间赋予不同颜色
        colors = plt.cm.viridis(np.array(all_confidences))
        
        plt.scatter(all_confidences, all_ious, c=colors, alpha=0.6, edgecolors='black')
        
        plt.title('置信度与IoU关系散点图')
        plt.xlabel('置信度')
        plt.ylabel('IoU值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='置信度')
        plt.tight_layout()
        plt.savefig(str(output_dir / "confidence_iou_scatter.png"), dpi=300)
        plt.close()
    
    # 7. 绘制真实目标数量与检测目标数量对比的散点图
    all_classes = sorted(set(list(all_true_counts.keys()) + list(all_pred_counts.keys())))
    true_values = np.array([all_true_counts.get(cls, 0) for cls in all_classes])
    pred_values = np.array([all_pred_counts.get(cls, 0) for cls in all_classes])
    
    if len(true_values) > 0 and len(pred_values) > 0:
        plt.figure(figsize=(10, 8))
        
        # 绘制对角线 (理想情况)
        max_val = max(np.max(true_values), np.max(pred_values))
        diag_line = np.linspace(0, max_val, 100)
        plt.plot(diag_line, diag_line, 'k--', alpha=0.5, label='完美匹配')
        
        # 绘制散点
        for i, cls in enumerate(all_classes):
            plt.scatter(true_values[i], pred_values[i], s=100, 
                      label=cls, edgecolors='black')
            plt.annotate(cls, (true_values[i], pred_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('真实目标数量与检测目标数量对比')
        plt.xlabel('真实目标数量')
        plt.ylabel('检测目标数量')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加每个类别的准确率标注
        # 计算比率
        accuracy = []
        for i in range(len(all_classes)):
            if true_values[i] > 0:
                ratio = pred_values[i] / true_values[i]
                accuracy.append((all_classes[i], ratio))
        
        # 设置图例，但删除类别标签以避免拥挤
        #plt.legend(loc='upper left')
        
        # 显示准确率信息
        info_text = "类别检出比率:\n"
        for cls, ratio in accuracy:
            info_text += f"{cls}: {ratio:.2f}\n"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "detection_count_scatter.png"), dpi=300)
        plt.close()
        
        # 8. 绘制阈值变化对精确率和召回率的影响曲线 (模拟数据)
        # 这里我们没有不同阈值下的数据，所以创建模拟数据展示效果
        thresholds = np.linspace(0.05, 0.95, 10)
        
        # 假设精确率随阈值增加而增加
        precision_curve = 0.5 + 0.5 * (1 - np.exp(-5 * thresholds))
        
        # 假设召回率随阈值增加而减少
        recall_curve = 0.9 * np.exp(-3 * thresholds) + 0.1
        
        # 计算F1分数
        f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(thresholds, precision_curve, 'b-', linewidth=2, label='精确率 (模拟)')
        plt.plot(thresholds, recall_curve, 'r-', linewidth=2, label='召回率 (模拟)')
        plt.plot(thresholds, f1_curve, 'g-', linewidth=2, label='F1分数 (模拟)')
        
        plt.axvline(x=0.25, color='black', linestyle='--', 
                   label='当前阈值 (0.25)')
        
        plt.title('置信度阈值对性能的影响 (模拟数据)')
        plt.xlabel('置信度阈值')
        plt.ylabel('性能指标')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(str(output_dir / "threshold_curve_simulation.png"), dpi=300)
        plt.close()
        
    # 9. 尝试创建3D柱状图，显示不同类别在不同评估指标上的表现
    if len(merged_class_stats) > 0:
        try:
            # 准备数据
            classes = list(merged_class_stats.keys())
            metrics_names = ['精确率', '召回率', 'F1分数']
            
            x = np.arange(len(classes))
            y = np.arange(len(metrics_names))
            
            # 创建网格
            xpos, ypos = np.meshgrid(x, y, indexing='ij')
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)
            
            # 设置柱状图的尺寸
            dx = 0.8 * np.ones_like(zpos)
            dy = 0.8 * np.ones_like(zpos)
            
            # 设置柱状图的高度
            dz = np.zeros_like(zpos)
            for i, class_name in enumerate(classes):
                dz[i*3] = merged_class_stats[class_name].get('precision', 0)
                dz[i*3+1] = merged_class_stats[class_name].get('recall', 0)
                dz[i*3+2] = merged_class_stats[class_name].get('f1_score', 0)
            
            # 设置柱状图的颜色
            colors = []
            for i in range(len(classes)):
                colors.extend(['r', 'g', 'b'])  # 红色：精确率，绿色：召回率，蓝色：F1分数
            
            # 创建3D图
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.6)
            
            # 设置坐标轴标签
            ax.set_xlabel('类别')
            ax.set_ylabel('评估指标')
            ax.set_zlabel('得分')
            
            # 设置刻度标签
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.set_yticks(y)
            ax.set_yticklabels(metrics_names)
            
            # 设置图表标题
            ax.set_title('各类别在不同评估指标上的表现')
            
            # 设置图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='r', markersize=10, label='精确率'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='g', markersize=10, label='召回率'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='b', markersize=10, label='F1分数')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "3d_performance_bars.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"创建3D柱状图时出错: {e}")
    
    # 10. 绘制箱线图，显示各类别IoU分布
    class_iou_data = {}
    for metrics in all_metrics:
        for class_name, stats in metrics['class_stats'].items():
            if 'iou_scores' in stats and stats['iou_scores']:
                if class_name not in class_iou_data:
                    class_iou_data[class_name] = []
                class_iou_data[class_name].extend(stats['iou_scores'])
    
    if class_iou_data:
        plt.figure(figsize=(12, 8))
        
        # 转换为pandas DataFrame进行绘图
        boxplot_data = []
        box_labels = []
        
        for class_name, iou_scores in class_iou_data.items():
            if iou_scores:  # 确保有数据
                boxplot_data.append(iou_scores)
                box_labels.append(class_name)
        
        if boxplot_data:
            plt.boxplot(boxplot_data, labels=box_labels, patch_artist=True)
            
            # 添加数据点
            for i, data in enumerate(boxplot_data):
                # 在数据点的位置添加少量随机抖动，以便更好地可视化
                x = np.random.normal(i+1, 0.04, size=len(data))
                plt.scatter(x, data, alpha=0.4, color='black', s=5)
            
            plt.title('各类别IoU分布箱线图')
            plt.ylabel('IoU值')
            plt.xlabel('类别')
            plt.grid(True, linestyle='--', axis='y', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "iou_boxplot.png"), dpi=300)
            plt.close()
    
    # 11. 绘制交互式HTML散点图（将保存为静态图像）
    if all_confidences and all_ious:
        plt.figure(figsize=(10, 8))
        
        # 根据IoU质量绘制散点图，颜色表示质量级别
        colors = []
        sizes = []
        for iou in all_ious:
            if iou < 0.5:  # 较差的IoU
                colors.append('red')
                sizes.append(50)
            elif iou < 0.7:  # 中等的IoU
                colors.append('orange')
                sizes.append(100)
            else:  # 良好的IoU
                colors.append('green')
                sizes.append(150)
        
        plt.scatter(all_confidences, all_ious, c=colors, s=sizes, alpha=0.6, edgecolors='black')
        
        plt.title('置信度与IoU关系质量图')
        plt.xlabel('置信度')
        plt.ylabel('IoU值')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 创建图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='IoU < 0.5'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='0.5 ≤ IoU < 0.7'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='IoU ≥ 0.7')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "quality_confidence_iou_scatter.png"), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="使用训练好的YOLOv8模型进行预测")
    parser.add_argument('--img_folder', type=str, help='要预测的图像文件夹路径')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--samples', type=int, default=20, help='随机抽取的样本数量')
    parser.add_argument('--all', action='store_true', help='处理所有图像，不随机抽样')
    args = parser.parse_args()
    
    # 使用Path处理路径
    base_dir = Path(__file__).resolve().parent.parent.parent
    yolo_dir = base_dir / "yolo_code"
    
    # 设置目录
    config_path = yolo_dir / "config" / "insects.yaml"
    results_dir = yolo_dir / "results"
    models_dir = results_dir / "models"
    predictions_base_dir = results_dir / "predictions"
    logs_dir = results_dir / "logs"
    
    # 创建test子目录
    test_dir = predictions_base_dir / "test"
    test_dir.mkdir(exist_ok=True, parents=True)
    # 创建带时间戳的test子目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_test_dir = test_dir / timestamp
    current_test_dir.mkdir(exist_ok=True, parents=True)
    
    # 设置输出目录为带时间戳的test子目录
    output_dir = current_test_dir
    
    # 确保目录存在
    for directory in [logs_dir, output_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # 尝试加载本地字体或项目根目录下的字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 设置日志
    logger = setup_logging(logs_dir)
    logger.info(f"系统环境: {platform.system()} {platform.release()}")
    logger.info(f"预测输出目录: {output_dir}")
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        class_names = config['names']
        logger.info(f"加载类别信息: {class_names}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 加载模型
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.warning(f"指定的模型文件不存在: {model_path}, 将尝试查找其他模型")
            model_path = None
    
    if not model_path:
        # 直接指定默认模型路径
        default_model_path = results_dir / "train11" / "weights" / "best.pt"
        if default_model_path.exists():
            model_path = default_model_path
            logger.info(f"使用指定的默认模型: {model_path}")
        else:
            # 尝试加载models目录中的best_model.pt
            fallback_model_path = models_dir / "best_model.pt"
            if fallback_model_path.exists():
                model_path = fallback_model_path
                logger.info(f"使用默认模型: {model_path}")
            else:
                # 尝试在最新训练目录中查找
                latest_train_dir = find_latest_train_dir(results_dir)
                if latest_train_dir:
                    best_model = latest_train_dir / "weights" / "best.pt"
                    if best_model.exists():
                        model_path = best_model
                        logger.info(f"使用最新训练目录中的模型: {model_path}")
                
                # 如果仍然没找到，尝试查找最新的模型
                if not model_path:
                    model_path = find_latest_model(yolo_dir)
                    if model_path:
                        logger.info(f"找到最新模型: {model_path}")
                    else:
                        logger.error("未找到可用的模型文件")
                        return
    
    logger.info(f"加载模型: {model_path}")
    # 修改模型加载部分
    try:
        # 尝试使用YOLOv10加载方式（如果您有特定的导入方法）
        try:
            # 如果有专门的YOLOv10导入方式
            from ultralytics.models import YOLO as YOLOv10
            model = YOLOv10(model_path)
            logger.info("使用YOLOv10模型加载成功")
        except (ImportError, AttributeError):
            # 退回到通用YOLO导入
            from ultralytics.models import YOLO
            model = YOLO(model_path)
            logger.info("使用通用YOLO接口加载模型成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return
    
    # 确定数据集路径
    test_images_dir = yolo_dir / "data" / "images" / "test"
    test_labels_dir = yolo_dir / "data" / "labels" / "test"
    val_images_dir = yolo_dir / "data" / "images" / "val"
    val_labels_dir = yolo_dir / "data" / "labels" / "val"
    
    # 收集所有图像文件和对应的标签文件
    test_image_files = []
    test_label_files = []
    if test_images_dir.exists():
        for ext in ['.jpg', '.jpeg', '.png']:
            for img_file in test_images_dir.glob(f'*{ext}'):
                base_name = img_file.stem
                label_file = test_labels_dir / f"{base_name}.txt"
                if label_file.exists():
                    test_image_files.append(img_file)
                    test_label_files.append(label_file)
    
    val_image_files = []
    val_label_files = []
    if val_images_dir.exists():
        for ext in ['.jpg', '.jpeg', '.png']:
            for img_file in val_images_dir.glob(f'*{ext}'):
                base_name = img_file.stem
                label_file = val_labels_dir / f"{base_name}.txt"
                if label_file.exists():
                    val_image_files.append(img_file)
                    val_label_files.append(label_file)
    
    # 合并测试集和验证集
    all_image_files = test_image_files + val_image_files
    all_label_files = test_label_files + val_label_files
    
    # 如果用户指定了图像文件夹
    if args.img_folder:
        img_folder = Path(args.img_folder)
        if img_folder.exists():
            # 清空之前的列表
            all_image_files = []
            all_label_files = []
            
            # 收集指定文件夹中的图像
            for ext in ['.jpg', '.jpeg', '.png']:
                for img_file in img_folder.glob(f'*{ext}'):
                    base_name = img_file.stem
                    # 尝试在多个可能的位置查找标签文件
                    label_file = None
                    
                    # 1. 在同级的labels目录中查找
                    labels_dir = img_folder.parent / "labels" / img_folder.name
                    if (labels_dir / f"{base_name}.txt").exists():
                        label_file = labels_dir / f"{base_name}.txt"
                    
                    # 2. 在同级目录中查找
                    elif (img_folder.parent / "labels" / f"{base_name}.txt").exists():
                        label_file = img_folder.parent / "labels" / f"{base_name}.txt"
                    
                    # 3. 在根目录的labels目录中查找
                    elif (yolo_dir / "data" / "labels" / "test" / f"{base_name}.txt").exists():
                        label_file = yolo_dir / "data" / "labels" / "test" / f"{base_name}.txt"
                    
                    # 4. 在图像同目录下查找
                    elif (img_folder / f"{base_name}.txt").exists():
                        label_file = img_folder / f"{base_name}.txt"
                    
                    # 如果找到标签文件，添加到列表
                    if label_file:
                        all_image_files.append(img_file)
                        all_label_files.append(label_file)
                    else:
                        logger.warning(f"未找到图像 {img_file} 的标签文件，将跳过该图像")
            
            logger.info(f"在指定文件夹中找到 {len(all_image_files)} 个有对应标签的图像")
        else:
            logger.error(f"指定的图像文件夹不存在: {args.img_folder}")
            return
    
    if not all_image_files:
        logger.error("未找到任何带标签的图像文件")
        return
    
    # 选择要处理的图像
    if args.all:
        # 处理所有图像
        selected_images = all_image_files
        selected_labels = all_label_files
    else:
        # 随机抽取指定数量的样本
        sample_count = min(args.samples, len(all_image_files))
        sample_indices = random.sample(range(len(all_image_files)), sample_count)
        
        selected_images = [all_image_files[i] for i in sample_indices]
        selected_labels = [all_label_files[i] for i in sample_indices]
    
    logger.info(f"将处理 {len(selected_images)} 个图像")
    
    # 创建多个子目录存放不同类型的结果
    comparison_dir = output_dir / "comparisons"
    heatmap_dir = output_dir / "heatmaps"
    stats_dir = output_dir / "statistics"
    merged_dir = output_dir / "merged"
    advanced_viz_dir = output_dir / "advanced_visualizations"
    
    for directory in [comparison_dir, heatmap_dir, stats_dir, merged_dir, advanced_viz_dir]:
        directory.mkdir(exist_ok=True)
    
    # 处理每个图像
    all_metrics = []
    all_true_counts = Counter()
    all_pred_counts = Counter()
    
    for i, (img_path, label_path) in enumerate(zip(selected_images, selected_labels)):
        logger.info(f"处理图像 {i+1}/{len(selected_images)}: {img_path.name}")
        
        # 预测并与真实标签对比
        metrics, true_counts, pred_counts = predict_with_ground_truth(
            model, 
            img_path, 
            label_path, 
            class_names, 
            output_dir,
            conf_threshold=args.conf
        )
        
        if metrics:
            all_metrics.append(metrics)
            # 更新总体统计数据
            for cls, count in true_counts.items():
                all_true_counts[cls] += count
            for cls, count in pred_counts.items():
                all_pred_counts[cls] += count
            
            # 移动生成的图像到对应子目录
            for file in output_dir.glob(f"compare_{img_path.stem}*"):
                shutil.move(str(file), str(comparison_dir / file.name))
            
            for file in output_dir.glob(f"heatmap_{img_path.stem}*"):
                shutil.move(str(file), str(heatmap_dir / file.name))
                
            for file in output_dir.glob(f"stats_compare_{img_path.stem}*"):
                shutil.move(str(file), str(stats_dir / file.name))
                
            for file in output_dir.glob(f"horiz_stats_compare_{img_path.stem}*"):
                shutil.move(str(file), str(stats_dir / file.name))
                
            for file in output_dir.glob(f"confidence_dist_{img_path.stem}*"):
                shutil.move(str(file), str(stats_dir / file.name))
                
            for file in output_dir.glob(f"iou_dist_{img_path.stem}*"):
                shutil.move(str(file), str(stats_dir / file.name))
                
            for file in output_dir.glob(f"merged_{img_path.stem}*"):
                shutil.move(str(file), str(merged_dir / file.name))
                
            # 创建一个高级可视化目录，用于存放3D图表等
            img_advanced_dir = advanced_viz_dir / img_path.stem
            img_advanced_dir.mkdir(exist_ok=True)
            
            # 生成单张图像的高级可视化
            try:
                if metrics['class_stats']:
                    # 类别性能雷达图
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, polar=True)
                    
                    categories = ['精确率', '召回率', 'F1分数']
                    N = len(categories)
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # 闭合雷达图
                    
                    for i, (class_name, stats) in enumerate(metrics['class_stats'].items()):
                        values = [
                            stats.get('precision', 0),
                            stats.get('recall', 0),
                            stats.get('f1_score', 0)
                        ]
                        values += values[:1]  # 闭合雷达图
                        
                        ax.plot(angles, values, linewidth=2, linestyle='solid', label=class_name)
                        ax.fill(angles, values, alpha=0.1)
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
                    ax.set_ylim(0, 1)
                    plt.title(f'图像 {img_path.stem} 类别性能雷达图')
                    plt.legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(str(img_advanced_dir / "class_radar.png"), dpi=300)
                    plt.close()
                    
                    # 创建类别性能热力图
                    classes = list(metrics['class_stats'].keys())
                    metrics_list = ['precision', 'recall', 'f1_score', 'true_positives', 'false_positives', 'false_negatives']
                    
                    # 准备热力图数据
                    heatmap_data = []
                    for cls in classes:
                        row = []
                        for metric in metrics_list:
                            row.append(metrics['class_stats'][cls].get(metric, 0))
                        heatmap_data.append(row)
                    
                    plt.figure(figsize=(12, len(classes) * 0.5 + 3))
                    sns.heatmap(heatmap_data, annot=True, fmt=".2f", 
                              xticklabels=metrics_list,
                              yticklabels=classes,
                              cmap="YlGnBu")
                    plt.title(f'图像 {img_path.stem} 类别性能热力图')
                    plt.tight_layout()
                    plt.savefig(str(img_advanced_dir / "class_metrics_heatmap.png"), dpi=300)
                    plt.close()
            except Exception as e:
                logger.error(f"生成图像 {img_path.stem} 的高级可视化时出错: {e}")
                traceback.print_exc()
    
    # 计算总体指标并保存
    if all_metrics:
        overall_stats = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'true_positives': sum([m['true_positives'] for m in all_metrics]),
            'false_positives': sum([m['false_positives'] for m in all_metrics]),
            'false_negatives': sum([m['false_negatives'] for m in all_metrics]),
        }
        
        # 合并所有metrics中的class_stats
        merged_class_stats = {}
        for metrics in all_metrics:
            for class_name, stats in metrics['class_stats'].items():
                if class_name not in merged_class_stats:
                    merged_class_stats[class_name] = {
                        'true': 0,
                        'predicted': 0,
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'iou_scores': [],
                    }
                
                merged_class_stats[class_name]['true'] += stats.get('true', 0)
                merged_class_stats[class_name]['predicted'] += stats.get('predicted', 0)
                merged_class_stats[class_name]['true_positives'] += stats.get('true_positives', 0)
                merged_class_stats[class_name]['false_positives'] += stats.get('false_positives', 0)
                merged_class_stats[class_name]['false_negatives'] += stats.get('false_negatives', 0)
                merged_class_stats[class_name]['iou_scores'].extend(stats.get('iou_scores', []))
        
        # 计算每个类别的精确率、召回率和F1分数
        for class_name, stats in merged_class_stats.items():
            if stats['true_positives'] + stats['false_positives'] > 0:
                stats['precision'] = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
            else:
                stats['precision'] = 0
            
            if stats['true_positives'] + stats['false_negatives'] > 0:
                stats['recall'] = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
            else:
                stats['recall'] = 0
            
            if stats['precision'] + stats['recall'] > 0:
                stats['f1_score'] = 2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall'])
            else:
                stats['f1_score'] = 0
        
        # 保存总体统计数据
        summary = {
            'model_path': str(model_path),
            'sample_count': len(selected_images),
            'overall_stats': overall_stats,
            'class_stats': {k: {sk: sv for sk, sv in v.items() if sk != 'iou_scores'} for k, v in merged_class_stats.items()},
            'true_counts': dict(all_true_counts),
            'pred_counts': dict(all_pred_counts),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        logger.info(f"总体精确率: {overall_stats['precision']:.4f}, 总体召回率: {overall_stats['recall']:.4f}, 总体F1分数: {overall_stats['f1_score']:.4f}")
        
        # 创建统计摘要的美观HTML报告
        try:
            # 生成HTML报告
            html_report = generate_html_report(summary, all_metrics, output_dir)
            with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
                f.write(html_report)
            logger.info(f"生成HTML报告: {output_dir / 'report.html'}")
        except Exception as e:
            logger.error(f"生成HTML报告时出错: {e}")
            traceback.print_exc()
        
        # 绘制高级可视化
        try:
            plot_advanced_visualizations(all_metrics, all_true_counts, all_pred_counts, merged_class_stats, class_names, advanced_viz_dir)
            logger.info(f"生成高级可视化图表到目录: {advanced_viz_dir}")
            
            # 额外创建更多可视化图表
            generate_additional_visualizations(all_metrics, merged_class_stats, class_names, advanced_viz_dir)
            logger.info("生成额外的可视化图表")
        except Exception as e:
            logger.error(f"生成高级可视化时出错: {e}")
            traceback.print_exc()
    
    logger.info(f"预测完成，输出保存到: {output_dir}")

def generate_html_report(summary, all_metrics, output_dir):
    """生成HTML报告"""
    model_path = summary['model_path']
    sample_count = summary['sample_count']
    overall_stats = summary['overall_stats']
    class_stats = summary['class_stats']
    true_counts = summary['true_counts']
    pred_counts = summary['pred_counts']
    timestamp = summary['timestamp']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>目标检测评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric {{ font-weight: bold; }}
            .good {{ color: green; }}
            .average {{ color: orange; }}
            .poor {{ color: red; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
            .gallery {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .gallery img {{ width: 300px; height: auto; object-fit: cover; }}
        </style>
    </head>
    <body>
        <h1>目标检测评估报告</h1>
        <div class="summary">
            <p><strong>模型路径:</strong> {model_path}</p>
            <p><strong>评估样本数量:</strong> {sample_count}</p>
            <p><strong>评估时间:</strong> {timestamp}</p>
        </div>
        
        <h2>总体性能</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>值</th>
            </tr>
            <tr>
                <td>精确率 (Precision)</td>
                <td class="metric {get_performance_class(overall_stats['precision'])}">{overall_stats['precision']:.4f}</td>
            </tr>
            <tr>
                <td>召回率 (Recall)</td>
                <td class="metric {get_performance_class(overall_stats['recall'])}">{overall_stats['recall']:.4f}</td>
            </tr>
            <tr>
                <td>F1分数</td>
                <td class="metric {get_performance_class(overall_stats['f1_score'])}">{overall_stats['f1_score']:.4f}</td>
            </tr>
            <tr>
                <td>真阳性 (True Positives)</td>
                <td>{overall_stats['true_positives']}</td>
            </tr>
            <tr>
                <td>假阳性 (False Positives)</td>
                <td>{overall_stats['false_positives']}</td>
            </tr>
            <tr>
                <td>假阴性 (False Negatives)</td>
                <td>{overall_stats['false_negatives']}</td>
            </tr>
        </table>
        
        <h2>类别性能</h2>
        <table>
            <tr>
                <th>类别</th>
                <th>真实目标数</th>
                <th>预测目标数</th>
                <th>精确率</th>
                <th>召回率</th>
                <th>F1分数</th>
                <th>真阳性</th>
                <th>假阳性</th>
                <th>假阴性</th>
            </tr>
    """
    
    # 添加类别统计
    for class_name, stats in sorted(class_stats.items()):
        html += f"""
            <tr>
                <td>{class_name}</td>
                <td>{stats['true']}</td>
                <td>{stats['predicted']}</td>
                <td class="{get_performance_class(stats.get('precision', 0))}">{stats.get('precision', 0):.4f}</td>
                <td class="{get_performance_class(stats.get('recall', 0))}">{stats.get('recall', 0):.4f}</td>
                <td class="{get_performance_class(stats.get('f1_score', 0))}">{stats.get('f1_score', 0):.4f}</td>
                <td>{stats['true_positives']}</td>
                <td>{stats['false_positives']}</td>
                <td>{stats['false_negatives']}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>目标数量对比</h2>
        <table>
            <tr>
                <th>类别</th>
                <th>真实目标数</th>
                <th>预测目标数</th>
                <th>比率</th>
            </tr>
    """
    
    # 添加目标数量对比
    for class_name in sorted(set(list(true_counts.keys()) + list(pred_counts.keys()))):
        true_count = true_counts.get(class_name, 0)
        pred_count = pred_counts.get(class_name, 0)
        ratio = pred_count / true_count if true_count > 0 else float('inf')
        ratio_display = f"{ratio:.2f}" if ratio != float('inf') else "∞"
        
        html += f"""
            <tr>
                <td>{class_name}</td>
                <td>{true_count}</td>
                <td>{pred_count}</td>
                <td>{ratio_display}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>可视化示例</h2>
    """
    
    # 添加可视化图像链接
    advanced_viz_dir = output_dir / "advanced_visualizations"
    comparison_dir = output_dir / "comparisons"
    
    html += """
        <h3>高级可视化</h3>
        <div class="gallery">
    """
    
    for img_path in advanced_viz_dir.glob("*.png"):
        html += f"""
            <img src="{img_path.relative_to(output_dir)}" alt="{img_path.stem}" title="{img_path.stem}">
        """
    
    html += """
        </div>
        
        <h3>样本预测对比</h3>
        <div class="gallery">
    """
    
    # 添加最多5张比较图片
    for i, img_path in enumerate(comparison_dir.glob("compare_*.jpg")):
        if i >= 5:
            break
        html += f"""
            <img src="{img_path.relative_to(output_dir)}" alt="{img_path.stem}" title="{img_path.stem}">
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def get_performance_class(value):
    """根据性能值返回CSS类"""
    if value >= 0.7:
        return "good"
    elif value >= 0.5:
        return "average"
    else:
        return "poor"

def generate_additional_visualizations(all_metrics, merged_class_stats, class_names, output_dir):
    """生成额外的可视化图表"""
    # 1. 创建更丰富的混淆矩阵可视化
    if merged_class_stats:
        # 提取类别名称和标签
        classes = sorted(merged_class_stats.keys())
        
        # 创建混淆矩阵数据
        cm = np.zeros((len(classes), len(classes)))
        
        # 对角线填充真阳性
        for i, class_name in enumerate(classes):
            cm[i, i] = merged_class_stats[class_name]['true_positives']
        
        # 使用更美观的颜色图
        plt.figure(figsize=(12, 10))
        ax = plt.subplot()
        
        # 使用Seaborn创建更美观的热力图
        sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=classes, yticklabels=classes, 
                  cmap='YlGnBu', linewidths=0.5, ax=ax)
        
        # 添加标题和标签
        plt.title('目标检测混淆矩阵 (真阳性)', fontsize=14)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        
        # 旋转x轴标签，以防重叠
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "enhanced_confusion_matrix.png"), dpi=300)
        plt.close()
        
        # 2. 绘制排名靠前的IoU分数分布
        all_ious = []
        
        for class_name, stats in merged_class_stats.items():
            if 'iou_scores' in stats and stats['iou_scores']:
                all_ious.extend([(class_name, iou) for iou in stats['iou_scores']])
        
        if all_ious:
            # 按IoU分数排序
            all_ious.sort(key=lambda x: x[1], reverse=True)
            
            # 取前50个或全部
            top_ious = all_ious[:min(50, len(all_ious))]
            
            plt.figure(figsize=(12, 6))
            
            classes_top_iou = [item[0] for item in top_ious]
            iou_values = [item[1] for item in top_ious]
            
            # 为每个类别分配一个颜色
            unique_classes = list(set(classes_top_iou))
            color_map = {}
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
            
            for i, cls in enumerate(unique_classes):
                color_map[cls] = colors[i]
            
            # 创建柱状图，颜色表示类别
            bars = plt.bar(range(len(top_ious)), iou_values, alpha=0.7)
            
            # 设置颜色
            for i, bar in enumerate(bars):
                bar.set_color(color_map[classes_top_iou[i]])
            
            plt.title('前50个最高IoU分数分布')
            plt.xlabel('排名')
            plt.ylabel('IoU分数')
            plt.ylim(0, 1.05)
            
            # 创建图例
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_map[cls], markersize=8, 
                                     label=cls) for cls in unique_classes]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "top_iou_scores.png"), dpi=300)
            plt.close()
    
    # 3. 生成综合展示性能的玫瑰图
    if merged_class_stats:
        plt.figure(figsize=(12, 10))
        
        # 提取数据
        classes = sorted(merged_class_stats.keys())
        precision_values = [merged_class_stats[cls].get('precision', 0) for cls in classes]
        recall_values = [merged_class_stats[cls].get('recall', 0) for cls in classes]
        f1_values = [merged_class_stats[cls].get('f1_score', 0) for cls in classes]
        
        # 角度值 (均匀分布)
        N = len(classes)
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        
        # 绘制极坐标图
        ax = plt.subplot(111, polar=True)
        
        # 为了使图表更清晰，我们为不同指标绘制不同半径的条形图
        width = 2 * np.pi / N * 0.8  # 每个扇形的宽度
        
        # 精确率条形图
        bars1 = ax.bar(theta, precision_values, width=width, alpha=0.6, 
                      label='精确率', bottom=0.0)
        
        # 召回率条形图 (外圈)
        bars2 = ax.bar(theta, recall_values, width=width, alpha=0.6, 
                      label='召回率', bottom=0.0, color='orange')
        
        # F1分数条形图 (最外圈)
        bars3 = ax.bar(theta, f1_values, width=width, alpha=0.6, 
                      label='F1分数', bottom=0.0, color='green')
        
        # 设置坐标轴
        ax.set_xticks(theta)
        ax.set_xticklabels(classes)
        
        # 设置刻度标签
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_ylim(0, 1)
        
        # 添加图例和标题
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('类别性能玫瑰图', size=15)
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "performance_rose.png"), dpi=300)
        plt.close()
    
    # 4. 绘制精确率-召回率-F1分数的三维散点图
    if merged_class_stats:
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 提取数据
            classes = list(merged_class_stats.keys())
            precision_values = [merged_class_stats[cls].get('precision', 0) for cls in classes]
            recall_values = [merged_class_stats[cls].get('recall', 0) for cls in classes]
            f1_values = [merged_class_stats[cls].get('f1_score', 0) for cls in classes]
            
            # 点的大小与类别中的实例数量成正比
            sizes = [merged_class_stats[cls]['true'] * 20 for cls in classes]
            
            # 设置颜色为热图
            colors = plt.cm.viridis(np.array(f1_values))
            
            # 绘制3D散点图
            scatter = ax.scatter(precision_values, recall_values, f1_values,
                               s=sizes, c=colors, marker='o', alpha=0.6)
            
            # 添加类别标签
            for i, cls in enumerate(classes):
                ax.text(precision_values[i], recall_values[i], f1_values[i], 
                      cls, fontsize=8)
            
            # 设置坐标轴标签
            ax.set_xlabel('精确率')
            ax.set_ylabel('召回率')
            ax.set_zlabel('F1分数')
            
            # 设置坐标轴范围
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            
            # 添加网格
            ax.grid(True)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label('F1分数')
            
            plt.title('类别性能三维散点图')
            plt.tight_layout()
            plt.savefig(str(output_dir / "3d_performance_scatter.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"创建3D散点图时出错: {e}")
            traceback.print_exc()
            
    # 5. 百分比堆叠条形图
    if merged_class_stats:
        plt.figure(figsize=(14, 8))
        
        # 提取数据
        classes = sorted(merged_class_stats.keys())
        tp = [merged_class_stats[cls]['true_positives'] for cls in classes]
        fp = [merged_class_stats[cls]['false_positives'] for cls in classes]
        fn = [merged_class_stats[cls]['false_negatives'] for cls in classes]
        
        # 计算每个类别的总样本数
        totals = [tp[i] + fp[i] + fn[i] for i in range(len(classes))]
        
        # 计算百分比
        tp_percent = [tp[i]/totals[i]*100 if totals[i] > 0 else 0 for i in range(len(classes))]
        fp_percent = [fp[i]/totals[i]*100 if totals[i] > 0 else 0 for i in range(len(classes))]
        fn_percent = [fn[i]/totals[i]*100 if totals[i] > 0 else 0 for i in range(len(classes))]
        
        # 创建百分比堆叠条形图
        ind = np.arange(len(classes))
        width = 0.7
        
        plt.bar(ind, tp_percent, width, label='真阳性', color='forestgreen')
        plt.bar(ind, fp_percent, width, bottom=tp_percent, label='假阳性', color='tomato')
        plt.bar(ind, fn_percent, width, bottom=[tp_percent[i] + fp_percent[i] for i in range(len(classes))], 
              label='假阴性', color='royalblue')
        
        plt.ylabel('百分比 (%)')
        plt.title('各类别检测结果百分比分布')
        plt.xticks(ind, classes, rotation=45, ha='right')
        plt.legend()
        
        # 添加总样本数标注
        for i, total in enumerate(totals):
            plt.text(i, 105, f'n={total}', ha='center', va='bottom', rotation=0,
                   fontsize=9, color='black')
        
        plt.ylim(0, 110)  # 为标注留出空间
        plt.tight_layout()
        plt.savefig(str(output_dir / "percentage_distribution.png"), dpi=300)
        plt.close()
        
    # 6. 创建镶嵌图，展示多个指标
    if merged_class_stats:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        
        # 左上: 精确率柱状图
        ax1 = fig.add_subplot(gs[0, 0])
        classes = sorted(merged_class_stats.keys())
        precision_values = [merged_class_stats[cls].get('precision', 0) for cls in classes]
        
        bars = ax1.bar(classes, precision_values, color='skyblue')
        # 为每个条添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', rotation=0,
                   fontsize=8)
        
        ax1.set_title('各类别精确率')
        ax1.set_ylim(0, 1.1)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 右上: 召回率柱状图
        ax2 = fig.add_subplot(gs[0, 1])
        recall_values = [merged_class_stats[cls].get('recall', 0) for cls in classes]
        
        bars = ax2.bar(classes, recall_values, color='lightcoral')
        # 为每个条添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', rotation=0,
                   fontsize=8)
        
        ax2.set_title('各类别召回率')
        ax2.set_ylim(0, 1.1)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 左下: 目标数量分布
        ax3 = fig.add_subplot(gs[1, 0])
        true_counts = [merged_class_stats[cls]['true'] for cls in classes]
        pred_counts = [merged_class_stats[cls]['predicted'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax3.bar(x - width/2, true_counts, width, label='真实标签', color='mediumseagreen')
        ax3.bar(x + width/2, pred_counts, width, label='预测结果', color='mediumpurple')
        
        ax3.set_title('各类别目标数量')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 右下: F1分数热力图
        ax4 = fig.add_subplot(gs[1, 1])
        f1_values = [merged_class_stats[cls].get('f1_score', 0) for cls in classes]
        
        # 创建热力图数据 (一维转二维)
        data = np.array(f1_values).reshape(1, -1)
        
        im = ax4.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('F1分数')
        
        # 设置热力图刻度和标签
        ax4.set_xticks(np.arange(len(classes)))
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.set_yticks([])  # 隐藏y轴刻度
        
        # 在每个单元格中添加数值
        for i in range(len(classes)):
            ax4.text(i, 0, f'{f1_values[i]:.2f}', 
                   ha='center', va='center', color='black',
                   fontsize=10)
        
        ax4.set_title('各类别F1分数热力图')
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "metrics_dashboard.png"), dpi=300)
        plt.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"运行时出错: {e}")
        traceback.print_exc()