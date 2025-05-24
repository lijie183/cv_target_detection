import os
import yaml
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
import argparse
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import matplotlib
import platform
import sys
import random
import logging
import traceback
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.metrics import confusion_matrix
import pandas as pd
import shutil

# 设置中文字体
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

def get_faster_rcnn_model(num_classes):
    # 加载Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # 替换分类器头以匹配我们的类别数量
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_tensor = torchvision.transforms.ToTensor()(image)
    return image, image_tensor

def predict_image(model, image_tensor, device, conf_threshold=0.5):
    # 进行预测
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
    
    # 提取预测结果
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    # 使用置信度阈值过滤结果
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    return boxes, labels, scores

def draw_predictions(image, boxes, labels, scores, class_names):
    # 绘制预测结果
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # 随机生成不同类别的颜色
    colors = {}
    for i in range(1, len(class_names)):
        colors[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # 绘制每个预测框
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        color = colors.get(label, (255, 0, 0))
        
        # 绘制框
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline=color,
            width=3
        )
        
        # 绘制标签
        label_text = f"{class_names[label]}: {score:.2f}"
        try:
            # 尝试获取字体
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        text_width, text_height = draw.textsize(label_text, font)
        draw.rectangle(
            [(box[0], box[1] - text_height - 4), (box[0] + text_width, box[1])],
            fill=color
        )
        draw.text((box[0], box[1] - text_height - 2), label_text, fill=(255, 255, 255), font=font)
    
    return image

def parse_xml(xml_path):
    """解析XML标注文件，获取真实标签信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'class_name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return objects

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算IoU
    iou = intersection / (area1 + area2 - intersection)
    return iou

def calculate_metrics(true_objects, pred_boxes, pred_labels, pred_scores, class_names, iou_threshold=0.5):
    """计算预测性能指标"""
    metrics = {
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'class_stats': {},
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'iou_scores': [],
        'confidence_scores': pred_scores.tolist()
    }
    
    # 初始化每个类别的统计
    for i, class_name in enumerate(class_names):
        if i > 0:  # 跳过背景类
            metrics['class_stats'][class_name] = {
                'true': 0,
                'predicted': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'iou_scores': []
            }
    
    # 统计真实标签中的类别数量
    for obj in true_objects:
        class_name = obj['class_name']
        if class_name in metrics['class_stats']:
            metrics['class_stats'][class_name]['true'] += 1
    
    # 统计预测标签中的类别数量
    for label in pred_labels:
        class_name = class_names[label]
        if class_name in metrics['class_stats']:
            metrics['class_stats'][class_name]['predicted'] += 1
    
    # 标记匹配的真实目标
    matched = [False] * len(true_objects)
    
    # 对每个预测框进行评估
    for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        class_name = class_names[pred_label]
        
        # 寻找最佳匹配的真实目标
        best_iou = 0
        best_match = -1
        for j, obj in enumerate(true_objects):
            if matched[j] or obj['class_name'] != class_name:
                continue
            
            iou = calculate_iou(pred_box, obj['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = j
        
        # 如果找到了有效匹配
        if best_iou >= iou_threshold:
            matched[best_match] = True
            metrics['true_positives'] += 1
            metrics['class_stats'][class_name]['true_positives'] += 1
            metrics['iou_scores'].append(best_iou)
            metrics['class_stats'][class_name]['iou_scores'].append(best_iou)
        else:
            metrics['false_positives'] += 1
            metrics['class_stats'][class_name]['false_positives'] += 1
    
    # 计算假阴性（未匹配的真实目标）
    for j, obj in enumerate(true_objects):
        if not matched[j]:
            class_name = obj['class_name']
            if class_name in metrics['class_stats']:
                metrics['false_negatives'] += 1
                metrics['class_stats'][class_name]['false_negatives'] += 1
    
    # 计算精确率和召回率
    if metrics['true_positives'] + metrics['false_positives'] > 0:
        metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
    if metrics['true_positives'] + metrics['false_negatives'] > 0:
        metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
    
    # 计算F1分数
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    
    # 计算每个类别的指标
    for class_name, stats in metrics['class_stats'].items():
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
    
    return metrics

def predict_with_ground_truth(model, img_path, xml_path, class_names, output_dir, device, conf_threshold=0.5):
    """对单张图片进行预测，并与真实标签比较"""
    try:
        img_path = Path(img_path)
        img_name = img_path.name
        base_name = img_path.stem
        
        # 加载图像
        original_image, image_tensor = load_image(img_path)
        width, height = original_image.size
        
        # 预测
        boxes, labels, scores = predict_image(model, image_tensor, device, conf_threshold)
        
        # 读取真实标签
        true_objects = []
        if xml_path.exists():
            true_objects = parse_xml(xml_path)
        
        # 绘制预测结果
        pred_image = original_image.copy()
        pred_image = draw_predictions(pred_image, boxes, labels, scores, class_names)
        
        # 绘制真实标签
        true_image = original_image.copy()
        draw = ImageDraw.Draw(true_image)
        
        # 为每个类别生成一个稳定的颜色
        class_colors = {}
        for i, name in enumerate(class_names):
            if i > 0:  # 跳过背景类
                class_colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        for obj in true_objects:
            class_name = obj['class_name']
            box = obj['bbox']
            color = class_colors.get(class_name, (0, 255, 0))
            
            # 绘制框
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])],
                outline=color,
                width=3
            )
            
            # 绘制标签
            label_text = class_name
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font = ImageFont.load_default()
                
            text_width, text_height = draw.textsize(label_text, font)
            draw.rectangle(
                [(box[0], box[1] - text_height - 4), (box[0] + text_width, box[1])],
                fill=color
            )
            draw.text((box[0], box[1] - text_height - 2), label_text, fill=(255, 255, 255), font=font)
        
        # 转换预测框和标签为更好处理的格式
        pred_results = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            pred_results.append({
                'class_id': label,
                'class_name': class_names[label],
                'bbox': box,
                'confidence': score
            })
        
        for obj in true_objects:
            obj['class_id'] = class_names.index(obj['class_name']) if obj['class_name'] in class_names else -1
        
        # 计算性能指标
        metrics = calculate_metrics(true_objects, boxes, labels, scores, class_names)
        
        # 创建对比可视化
        plt.figure(figsize=(15, 8))
        
        # 显示真实标签
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(true_image))
        plt.title('真实标签')
        plt.axis('off')
        
        # 显示预测结果
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(pred_image))
        plt.title('预测结果')
        plt.axis('off')
        
        plt.tight_layout()
        comparison_path = Path(output_dir) / f"compare_{base_name}.jpg"
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        
        # 绘制热力图对比
        plt.figure(figsize=(15, 8))
        
        # 创建真实标签热力图
        true_heatmap = np.zeros((height, width))
        for obj in true_objects:
            box = [int(x) for x in obj['bbox']]
            true_heatmap[box[1]:box[3], box[0]:box[2]] += 1
        
        # 创建预测热力图
        pred_heatmap = np.zeros((height, width))
        for box, score in zip(boxes, scores):
            box = [int(x) for x in box]
            pred_heatmap[box[1]:box[3], box[0]:box[2]] += score
        
        # 归一化热力图
        if true_heatmap.max() > 0:
            true_heatmap = true_heatmap / true_heatmap.max()
        if pred_heatmap.max() > 0:
            pred_heatmap = pred_heatmap / pred_heatmap.max()
        
        # 显示原图与热力图叠加
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(original_image))
        plt.imshow(true_heatmap, cmap='jet', alpha=0.5)
        plt.title('真实标签热力图')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(original_image))
        plt.imshow(pred_heatmap, cmap='jet', alpha=0.5)
        plt.title('预测结果热力图')
        plt.axis('off')
        
        plt.tight_layout()
        heatmap_path = Path(output_dir) / f"heatmap_{base_name}.jpg"
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        
        # 绘制目标统计对比图
        true_counts = {}
        for obj in true_objects:
            class_name = obj['class_name']
            if class_name not in true_counts:
                true_counts[class_name] = 0
            true_counts[class_name] += 1
        
        pred_counts = {}
        for label in labels:
            class_name = class_names[label]
            if class_name not in pred_counts:
                pred_counts[class_name] = 0
            pred_counts[class_name] += 1
        
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
            
            stats_path = Path(output_dir) / f"stats_{base_name}.png"
            plt.savefig(stats_path, dpi=300)
            plt.close()
        
        # 绘制置信度分布图
        if len(scores) > 0:
            plt.figure(figsize=(10, 6))
            
            # 按类别分组的置信度
            confidence_by_class = {}
            for label, score in zip(labels, scores):
                class_name = class_names[label]
                if class_name not in confidence_by_class:
                    confidence_by_class[class_name] = []
                confidence_by_class[class_name].append(score)
            
            for i, (class_name, confidences) in enumerate(confidence_by_class.items()):
                plt.hist(confidences, alpha=0.5, bins=10, 
                         label=f'{class_name} (均值={np.mean(confidences):.2f})')
            
            plt.title('预测置信度分布')
            plt.xlabel('置信度')
            plt.ylabel('计数')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            conf_path = Path(output_dir) / f"confidence_{base_name}.png"
            plt.savefig(conf_path, dpi=300)
            plt.close()
        
        # 保存原始图像与预测框的叠加图
        overlay_image = original_image.copy()
        draw = ImageDraw.Draw(overlay_image)
        
        # 真实框（绿色）
        for obj in true_objects:
            box = obj['bbox']
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])],
                outline=(0, 255, 0),
                width=2
            )
            draw.text((box[0], box[1] - 10), obj['class_name'], fill=(0, 255, 0))
        
        # 预测框（红色）
        for box, label, score in zip(boxes, labels, scores):
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])],
                outline=(255, 0, 0),
                width=2
            )
            draw.text((box[0], box[1] - 10), f"{class_names[label]}: {score:.2f}", fill=(255, 0, 0))
        
        overlay_path = Path(output_dir) / f"overlay_{base_name}.jpg"
        overlay_image.save(overlay_path)
        
        return metrics, true_counts, pred_counts
    
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {traceback.format_exc()}")
        return None, {}, {}

def find_latest_model(models_dir):
    """查找最新的模型文件"""
    model_files = list(Path(models_dir).glob("*.pth"))
    if not model_files:
        return None
    
    # 按修改时间排序，返回最新的
    return sorted(model_files, key=os.path.getmtime)[-1]

def plot_advanced_visualizations(all_metrics, all_true_counts, all_pred_counts, merged_class_stats, class_names, output_dir):
    """生成高级可视化图表"""
    # 1. 混淆矩阵
    if merged_class_stats:
        classes = sorted(merged_class_stats.keys())
        cm = np.zeros((len(classes), len(classes)))
        
        # 对角线是真阳性（只是一个近似）
        for i, class_name in enumerate(classes):
            if class_name in merged_class_stats:
                cm[i, i] = merged_class_stats[class_name]['true_positives']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.0f', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title('类别检测混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(str(output_dir / "confusion_matrix.png"), dpi=300)
        plt.close()
    
    # 2. 精确率-召回率散点图
    plt.figure(figsize=(10, 8))
    
    for class_name, stats in merged_class_stats.items():
        plt.scatter(stats.get('recall', 0), stats.get('precision', 0), 
                   label=class_name, s=100)
    
    # 平均值
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
    
    # 3. 气泡图
    plt.figure(figsize=(12, 8))
    
    x_values = []  # 召回率
    y_values = []  # 精确率
    sizes = []     # 气泡大小（样本数量）
    colors = []    # 气泡颜色（F1分数）
    labels = []    # 类别标签
    
    for class_name, stats in merged_class_stats.items():
        if stats['true'] > 0:
            x_values.append(stats.get('recall', 0))
            y_values.append(stats.get('precision', 0))
            sizes.append(stats['true'] * 30)
            colors.append(stats.get('f1_score', 0))
            labels.append(class_name)
    
    if x_values:
        scatter = plt.scatter(x_values, y_values, s=sizes, c=colors, 
                          alpha=0.6, cmap='viridis', edgecolors='black')
        
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
    
    # 4. 雷达图
    if merged_class_stats:
        categories = ['精确率', '召回率', 'F1分数']
        N = len(categories)
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图
        
        for i, (class_name, stats) in enumerate(merged_class_stats.items()):
            if i >= 10:  # 限制类别数量
                break
                
            values = [
                stats.get('precision', 0),
                stats.get('recall', 0),
                stats.get('f1_score', 0)
            ]
            values += values[:1]  # 闭合雷达图
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=class_name)
            ax.fill(angles, values, alpha=0.1)
        
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
        plt.ylim(0, 1)
        
        plt.title('类别性能雷达图')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(str(output_dir / "radar_chart.png"), dpi=300)
        plt.close()
    
    # 5. 堆叠条形图
    if merged_class_stats:
        data = []
        class_labels = []
        
        for class_name, stats in merged_class_stats.items():
            data.append([
                stats['true_positives'],
                stats['false_positives'],
                stats['false_negatives']
            ])
            class_labels.append(class_name)
        
        data = np.array(data)
        
        # 绘制非空类别
        non_empty_indices = [i for i, row in enumerate(data) if any(row)]
        
        if non_empty_indices:
            filtered_data = data[non_empty_indices]
            filtered_labels = [class_labels[i] for i in non_empty_indices]
            
            plt.figure(figsize=(12, 8))
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
            
            # 添加标签
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
    
    # 6. 真实目标与检测目标数量对比
    all_classes = sorted(set(list(all_true_counts.keys()) + list(all_pred_counts.keys())))
    true_values = np.array([all_true_counts.get(cls, 0) for cls in all_classes])
    pred_values = np.array([all_pred_counts.get(cls, 0) for cls in all_classes])
    
    if len(true_values) > 0 and len(pred_values) > 0:
        plt.figure(figsize=(10, 8))
        
        max_val = max(np.max(true_values), np.max(pred_values))
        diag_line = np.linspace(0, max_val, 100)
        plt.plot(diag_line, diag_line, 'k--', alpha=0.5, label='完美匹配')
        
        for i, cls in enumerate(all_classes):
            plt.scatter(true_values[i], pred_values[i], s=100, 
                      label=cls, edgecolors='black')
            plt.annotate(cls, (true_values[i], pred_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('真实目标数量与检测目标数量对比')
        plt.xlabel('真实目标数量')
        plt.ylabel('检测目标数量')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 计算比率
        accuracy = []
        for i in range(len(all_classes)):
            if true_values[i] > 0:
                ratio = pred_values[i] / true_values[i]
                accuracy.append((all_classes[i], ratio))
        
        info_text = "类别检出比率:\n"
        for cls, ratio in accuracy:
            info_text += f"{cls}: {ratio:.2f}\n"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "detection_count_scatter.png"), dpi=300)
        plt.close()

def generate_html_report(summary, output_dir):
    """生成HTML报告"""
    model_path = summary['model_path']
    sample_count = summary['sample_count']
    overall_stats = summary['overall_stats']
    class_stats = summary['class_stats']
    true_counts = summary['true_counts']
    pred_counts = summary['pred_counts']
    timestamp = summary['timestamp']
    
    # 定义性能等级的辅助函数
    def get_performance_class(value):
        if value >= 0.7:
            return "good"
        elif value >= 0.5:
            return "average"
        else:
            return "poor"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Faster R-CNN 目标检测评估报告</title>
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
        <h1>Faster R-CNN 目标检测评估报告</h1>
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

def main():
    parser = argparse.ArgumentParser(description="使用Faster R-CNN模型进行目标检测预测")
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--img_folder', type=str, help='要预测的图像文件夹路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--samples', type=int, default=20, help='随机抽取的样本数量')
    parser.add_argument('--all', action='store_true', help='处理所有图像，不随机抽样')
    args = parser.parse_args()
    
    # 使用Path处理路径
    base_dir = Path(__file__).resolve().parent.parent.parent
    frcnn_dir = base_dir / "frcnn_code"
    
    # 设置目录
    config_path = frcnn_dir / "config" / "insects.yaml"
    results_dir = frcnn_dir / "results"
    models_dir = results_dir / "models"
    predictions_base_dir = results_dir / "predictions"
    logs_dir = results_dir / "logs"
    
    # 创建test子目录并添加时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = predictions_base_dir / "test" / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建分类子目录
    comparison_dir = output_dir / "comparisons"
    heatmap_dir = output_dir / "heatmaps"
    stats_dir = output_dir / "statistics"
    advanced_viz_dir = output_dir / "advanced_visualizations"
    
    for directory in [comparison_dir, heatmap_dir, stats_dir, advanced_viz_dir, logs_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # 设置字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 设置日志
    logger = setup_logging(logs_dir)
    logger.info("启动Faster R-CNN预测脚本")
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        class_names = ['background'] + config['names']  # 背景类为第0类
        logger.info(f"加载类别信息: {class_names}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.warning(f"指定的模型文件不存在: {model_path}，将尝试查找其他模型")
            model_path = None
    
    if not model_path:
        # 尝试查找模型
        model_path = find_latest_model(models_dir)
        if model_path:
            logger.info(f"找到最新模型: {model_path}")
        else:
            logger.error("未找到可用的模型文件")
            return
    
    try:
        model = get_faster_rcnn_model(len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        logger.info(f"成功加载模型: {model_path}")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return
    
    # 确定数据集路径
    data_dir = frcnn_dir / "data"
    val_images_dir = data_dir / "insects" / "val" / "images"
    val_annotations_dir = data_dir / "insects" / "val" / "annotations" / "xmls"
    train_images_dir = data_dir / "insects" / "train" / "images"
    train_annotations_dir = data_dir / "insects" / "train" / "annotations" / "xmls"
    
    # 收集图像和标签文件
    image_files = []
    xml_files = []
    
    # 从验证集收集
    if val_images_dir.exists() and val_annotations_dir.exists():
        for img_file in val_images_dir.glob("*.jpeg"):
            xml_file = val_annotations_dir / f"{img_file.stem}.xml"
            if xml_file.exists():
                image_files.append(img_file)
                xml_files.append(xml_file)
    
    # 从训练集收集
    if len(image_files) < 50 and train_images_dir.exists() and train_annotations_dir.exists():
        for img_file in train_images_dir.glob("*.jpeg"):
            xml_file = train_annotations_dir / f"{img_file.stem}.xml"
            if xml_file.exists():
                image_files.append(img_file)
                xml_files.append(xml_file)
    
    # 如果用户指定了图像文件夹
    if args.img_folder:
        img_folder = Path(args.img_folder)
        if img_folder.exists():
            # 清空之前的列表
            image_files = []
            xml_files = []
            
            # 查找图像和对应的XML
            for img_file in img_folder.glob("*.jpeg"):
                # 尝试不同位置查找XML
                possible_xml_locations = [
                    img_folder.parent / "annotations" / "xmls" / f"{img_file.stem}.xml",
                    img_folder.parent / "annotations" / f"{img_file.stem}.xml",
                    img_folder / f"{img_file.stem}.xml"
                ]
                
                for xml_path in possible_xml_locations:
                    if xml_path.exists():
                        image_files.append(img_file)
                        xml_files.append(xml_path)
                        break
            
            logger.info(f"在指定文件夹中找到 {len(image_files)} 个有对应标签的图像")
        else:
            logger.error(f"指定的图像文件夹不存在: {args.img_folder}")
            return
    
    if not image_files:
        logger.error("未找到任何带标签的图像文件")
        return
    
    # 选择要处理的图像
    if args.all:
        selected_images = image_files
        selected_xmls = xml_files
    else:
        sample_count = min(args.samples, len(image_files))
        sample_indices = random.sample(range(len(image_files)), sample_count)
        
        selected_images = [image_files[i] for i in sample_indices]
        selected_xmls = [xml_files[i] for i in sample_indices]
    
    logger.info(f"将处理 {len(selected_images)} 个图像")
    
    # 处理每个图像
    all_metrics = []
    all_true_counts = Counter()
    all_pred_counts = Counter()
    
    for i, (img_path, xml_path) in enumerate(zip(selected_images, selected_xmls)):
        logger.info(f"处理图像 {i+1}/{len(selected_images)}: {img_path.name}")
        
        # 预测并与真实标签对比
        metrics, true_counts, pred_counts = predict_with_ground_truth(
            model,
            img_path,
            xml_path,
            class_names,
            output_dir,
            device,
            conf_threshold=args.conf
        )
        
        if metrics:
            all_metrics.append(metrics)
            # 更新总体统计数据
            for cls, count in true_counts.items():
                all_true_counts[cls] += count
            for cls, count in pred_counts.items():
                all_pred_counts[cls] += count
            
            # 移动生成的图像到对应的子目录
            for file in output_dir.glob(f"compare_{img_path.stem}*"):
                shutil.move(str(file), str(comparison_dir / file.name))
            
            for file in output_dir.glob(f"heatmap_{img_path.stem}*"):
                shutil.move(str(file), str(heatmap_dir / file.name))
                
            for file in output_dir.glob(f"stats_{img_path.stem}*"):
                shutil.move(str(file), str(stats_dir / file.name))
                
            for file in output_dir.glob(f"confidence_{img_path.stem}*"):
                shutil.move(str(file), str(stats_dir / file.name))
                
            for file in output_dir.glob(f"overlay_{img_path.stem}*"):
                shutil.move(str(file), str(comparison_dir / file.name))
    
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
        
        # 计算每个类别的指标
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
        
        # 生成高级可视化和HTML报告
        try:
            plot_advanced_visualizations(all_metrics, all_true_counts, all_pred_counts, merged_class_stats, class_names, advanced_viz_dir)
            logger.info(f"生成高级可视化图表到目录: {advanced_viz_dir}")
            
            html_report = generate_html_report(summary, output_dir)
            with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
                f.write(html_report)
            logger.info(f"生成HTML报告: {output_dir / 'report.html'}")
        except Exception as e:
            logger.error(f"生成报告时出错: {e}")
            traceback.print_exc()
            
    logger.info(f"预测完成，输出保存到: {output_dir}")

if __name__ == "__main__":
    main()