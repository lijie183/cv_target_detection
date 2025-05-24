import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
import yaml
import random
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics.models import YOLO
import cv2
import platform
import xml.etree.ElementTree as ET
import logging
import time
import traceback
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
import shutil
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

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
    
    log_file = os.path.join(log_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def load_yolo_model(model_path):
    """加载YOLOv8模型"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"加载YOLO模型失败: {e}")
        return None

def load_faster_rcnn_model(model_path, num_classes, device):
    """加载Faster R-CNN模型"""
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"加载Faster R-CNN模型失败: {e}")
        return None

def load_models(yolo_path, frcnn_path, num_classes, device):
    """加载两个模型"""
    yolo_model = load_yolo_model(yolo_path)
    frcnn_model = load_faster_rcnn_model(frcnn_path, num_classes, device)
    return yolo_model, frcnn_model

def read_xml_annotations(xml_path, class_names):
    """读取XML标注文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    
    # 获取图像大小
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 遍历所有目标
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_names:
            continue
        
        class_id = class_names.index(name)
        
        # 获取边界框坐标
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        annotations.append({
            'class_id': class_id,
            'class_name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'confidence': 1.0  # 真实标注的置信度设为1
        })
    
    return annotations

def predict_with_yolo(model, image_path, class_names, conf_threshold=0.25):
    """使用YOLOv8模型进行预测"""
    try:
        results = model(image_path, conf=conf_threshold)
        
        # 提取预测结果
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
        
        return pred_annotations
    
    except Exception as e:
        print(f"YOLO预测时出错: {e}")
        traceback.print_exc()
        return []

def predict_with_frcnn(model, image_path, class_names, device, conf_threshold=0.5):
    """使用Faster R-CNN模型进行预测"""
    try:
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = torchvision.transforms.ToTensor()(image)
        
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
        
        # 转换为统一格式
        pred_annotations = []
        
        for i in range(len(boxes)):
            box = boxes[i]
            label = int(labels[i])
            score = float(scores[i])
            
            if label < len(class_names):
                pred_annotations.append({
                    'class_id': label,
                    'class_name': class_names[label],
                    'bbox': [box[0], box[1], box[2], box[3]],
                    'confidence': score
                })
        
        return pred_annotations
    
    except Exception as e:
        print(f"Faster R-CNN预测时出错: {e}")
        traceback.print_exc()
        return []

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 返回IoU
    return intersection / union

def calculate_metrics(true_annotations, pred_annotations, iou_threshold=0.5):
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
        'confidence_scores': [],
        'average_precision': 0,
        'inference_time': 0  # 预留字段，在外部填充
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
    
    # 保存预测置信度
    metrics['confidence_scores'] = [pred['confidence'] for pred in pred_annotations]
    
    # 创建用于匹配的标记列表
    matched_true = [False] * len(true_annotations)
    true_positives = 0
    false_positives = 0
    
    # 类别统计
    class_stats = {}
    for anno in true_annotations:
        class_name = anno['class_name']
        if class_name not in class_stats:
            class_stats[class_name] = {'true': 0, 'predicted': 0, 'true_positives': 0, 
                                      'false_positives': 0, 'false_negatives': 0, 'iou_scores': []}
        class_stats[class_name]['true'] += 1
    
    for anno in pred_annotations:
        class_name = anno['class_name']
        if class_name not in class_stats:
            class_stats[class_name] = {'true': 0, 'predicted': 0, 'true_positives': 0, 
                                      'false_positives': 0, 'false_negatives': 0, 'iou_scores': []}
        class_stats[class_name]['predicted'] += 1
    
    # 对每个预测标注
    for pred in pred_annotations:
        matched = False
        best_iou = 0
        best_idx = -1
        
        # 找到最佳匹配的真实标注
        for idx, true in enumerate(true_annotations):
            if pred['class_id'] == true['class_id'] and not matched_true[idx]:
                iou = calculate_iou(pred['bbox'], true['bbox'])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    matched = True
        
        if matched:
            # 匹配成功 - 真阳性
            matched_true[best_idx] = True
            true_positives += 1
            class_name = pred['class_name']
            class_stats[class_name]['true_positives'] += 1
            metrics['iou_scores'].append(best_iou)
            class_stats[class_name]['iou_scores'].append(best_iou)
        else:
            # 匹配失败 - 假阳性
            false_positives += 1
            class_name = pred['class_name']
            class_stats[class_name]['false_positives'] += 1
    
    # 计算未匹配的真实标注 - 假阴性
    false_negatives = matched_true.count(False)
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
    
    # 计算每个类别的指标
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
    
    # 更新指标
    metrics['class_stats'] = class_stats
    metrics['true_positives'] = true_positives
    metrics['false_positives'] = false_positives
    metrics['false_negatives'] = false_negatives
    
    # 计算近似的平均精度AP
    if pred_annotations:
        # 对预测结果按置信度排序
        sorted_preds = sorted(pred_annotations, key=lambda x: x['confidence'], reverse=True)
        precisions = []
        recalls = []
        tp_cumsum = 0
        fp_cumsum = 0
        
        # 阈值从高到低，计算每个阈值下的精确率和召回率
        for i, pred in enumerate(sorted_preds):
            matched = False
            for true in true_annotations:
                if pred['class_id'] == true['class_id']:
                    iou = calculate_iou(pred['bbox'], true['bbox'])
                    if iou > iou_threshold:
                        matched = True
                        break
            
            if matched:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / len(true_annotations) if true_annotations else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 使用插值法计算平均精度
        # 我们使用11点插值法，它是一种简单的逼近
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if not recalls:
                continue
            precision_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            if precision_at_recall:
                ap += max(precision_at_recall) / 11
        
        metrics['average_precision'] = ap
    
    return metrics

def compare_predictions(image_path, xml_path, yolo_model, frcnn_model, class_names, device, output_dir, 
                        yolo_conf=0.25, frcnn_conf=0.5, iou_threshold=0.5):
    """比较两个模型在同一图像上的预测结果"""
    try:
        # 读取真实标注
        true_annotations = read_xml_annotations(xml_path, class_names)
        
        # 使用YOLOv8预测
        start_time = time.time()
        yolo_predictions = predict_with_yolo(yolo_model, image_path, class_names, yolo_conf)
        yolo_time = time.time() - start_time
        
        # 使用Faster R-CNN预测
        start_time = time.time()
        frcnn_predictions = predict_with_frcnn(frcnn_model, image_path, class_names, device, frcnn_conf)
        frcnn_time = time.time() - start_time
        
        # 计算指标
        yolo_metrics = calculate_metrics(true_annotations, yolo_predictions, iou_threshold)
        frcnn_metrics = calculate_metrics(true_annotations, frcnn_predictions, iou_threshold)
        
        # 添加推理时间
        yolo_metrics['inference_time'] = yolo_time
        frcnn_metrics['inference_time'] = frcnn_time
        
        # 准备可视化结果
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # 创建对比可视化
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # 绘制原始图像和真实标注
        axs[0].imshow(np.array(img))
        axs[0].set_title('原始标注')
        axs[0].axis('off')
        
        # 为不同类别分配不同颜色
        colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
        
        # 绘制真实标注框
        for anno in true_annotations:
            xmin, ymin, xmax, ymax = anno['bbox']
            cls_id = anno['class_id']
            color = colors[cls_id % len(colors)]
            
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                linewidth=2, edgecolor=color,
                                facecolor='none')
            axs[0].add_patch(rect)
            
            axs[0].text(xmin, ymin - 5, class_names[cls_id], 
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=8, color='white')
        
        # 绘制YOLOv8预测结果
        axs[1].imshow(np.array(img))
        axs[1].set_title(f'YOLOv8预测 (AP: {yolo_metrics["average_precision"]:.2f}, 时间: {yolo_time:.3f}s)')
        axs[1].axis('off')
        
        for anno in yolo_predictions:
            xmin, ymin, xmax, ymax = anno['bbox']
            cls_id = anno['class_id']
            conf = anno['confidence']
            color = colors[cls_id % len(colors)]
            
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                linewidth=2, edgecolor=color,
                                facecolor='none')
            axs[1].add_patch(rect)
            
            axs[1].text(xmin, ymin - 5, f'{class_names[cls_id]} {conf:.2f}', 
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=8, color='white')
        
        # 绘制Faster R-CNN预测结果
        axs[2].imshow(np.array(img))
        axs[2].set_title(f'Faster R-CNN预测 (AP: {frcnn_metrics["average_precision"]:.2f}, 时间: {frcnn_time:.3f}s)')
        axs[2].axis('off')
        
        for anno in frcnn_predictions:
            xmin, ymin, xmax, ymax = anno['bbox']
            cls_id = anno['class_id']
            conf = anno['confidence']
            color = colors[cls_id % len(colors)]
            
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                linewidth=2, edgecolor=color,
                                facecolor='none')
            axs[2].add_patch(rect)
            
            axs[2].text(xmin, ymin - 5, f'{class_names[cls_id]} {conf:.2f}', 
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=8, color='white')
        
        plt.tight_layout()
        
        # 保存对比图
        image_name = Path(image_path).stem
        output_path = Path(output_dir) / f"compare_{image_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建热力图对比
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # 显示原始图像
        axs[0].imshow(np.array(img))
        axs[0].set_title('原始图像')
        axs[0].axis('off')
        
        # 创建YOLOv8的热力图
        yolo_heatmap = np.zeros((height, width))
        for anno in yolo_predictions:
            xmin, ymin, xmax, ymax = map(int, anno['bbox'])
            conf = anno['confidence']
            yolo_heatmap[max(0, ymin):min(height, ymax), max(0, xmin):min(width, xmax)] += conf
        
        if yolo_heatmap.max() > 0:
            yolo_heatmap = yolo_heatmap / yolo_heatmap.max()
        
        # 创建Faster R-CNN的热力图
        frcnn_heatmap = np.zeros((height, width))
        for anno in frcnn_predictions:
            xmin, ymin, xmax, ymax = map(int, anno['bbox'])
            conf = anno['confidence']
            frcnn_heatmap[max(0, ymin):min(height, ymax), max(0, xmin):min(width, xmax)] += conf
        
        if frcnn_heatmap.max() > 0:
            frcnn_heatmap = frcnn_heatmap / frcnn_heatmap.max()
        
        # 显示热力图
        axs[1].imshow(np.array(img))
        im1 = axs[1].imshow(yolo_heatmap, cmap='jet', alpha=0.5)
        axs[1].set_title('YOLOv8热力图')
        axs[1].axis('off')
        
        axs[2].imshow(np.array(img))
        im2 = axs[2].imshow(frcnn_heatmap, cmap='jet', alpha=0.5)
        axs[2].set_title('Faster R-CNN热力图')
        axs[2].axis('off')
        
        plt.tight_layout()
        
        # 添加颜色条
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        
        # 保存热力图
        heatmap_path = Path(output_dir) / f"heatmap_{image_name}.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制类别分布对比
        yolo_classes = {}
        frcnn_classes = {}
        true_classes = {}
        
        for anno in true_annotations:
            class_name = anno['class_name']
            if class_name not in true_classes:
                true_classes[class_name] = 0
            true_classes[class_name] += 1
        
        for anno in yolo_predictions:
            class_name = anno['class_name']
            if class_name not in yolo_classes:
                yolo_classes[class_name] = 0
            yolo_classes[class_name] += 1
        
        for anno in frcnn_predictions:
            class_name = anno['class_name']
            if class_name not in frcnn_classes:
                frcnn_classes[class_name] = 0
            frcnn_classes[class_name] += 1
        
        # 合并所有类别
        all_classes = sorted(set(list(true_classes.keys()) + list(yolo_classes.keys()) + list(frcnn_classes.keys())))
        
        if all_classes:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(all_classes))
            width = 0.25
            
            true_counts = [true_classes.get(cls, 0) for cls in all_classes]
            yolo_counts = [yolo_classes.get(cls, 0) for cls in all_classes]
            frcnn_counts = [frcnn_classes.get(cls, 0) for cls in all_classes]
            
            ax.bar(x - width, true_counts, width, label='真实标签', color='tab:blue')
            ax.bar(x, yolo_counts, width, label='YOLOv8预测', color='tab:orange')
            ax.bar(x + width, frcnn_counts, width, label='Faster R-CNN预测', color='tab:green')
            
            ax.set_xticks(x)
            ax.set_xticklabels(all_classes, rotation=45, ha='right')
            ax.set_ylabel('目标数量')
            ax.set_title('各模型检测到的类别分布对比')
            ax.legend()
            
            plt.tight_layout()
            class_dist_path = Path(output_dir) / f"class_dist_{image_name}.png"
            plt.savefig(class_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 绘制置信度分布对比
        if yolo_predictions or frcnn_predictions:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            yolo_conf = [p['confidence'] for p in yolo_predictions]
            frcnn_conf = [p['confidence'] for p in frcnn_predictions]
            
            if yolo_conf:
                sns.kdeplot(yolo_conf, ax=ax, label=f'YOLOv8 (均值: {np.mean(yolo_conf):.2f})', shade=True)
            
            if frcnn_conf:
                sns.kdeplot(frcnn_conf, ax=ax, label=f'Faster R-CNN (均值: {np.mean(frcnn_conf):.2f})', shade=True)
            
            ax.set_xlabel('置信度')
            ax.set_ylabel('密度')
            ax.set_title('预测置信度分布对比')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            conf_dist_path = Path(output_dir) / f"conf_dist_{image_name}.png"
            plt.savefig(conf_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 返回指标，用于后续汇总分析
        return {
            'image_name': image_name,
            'yolo_metrics': yolo_metrics,
            'frcnn_metrics': frcnn_metrics,
            'true_count': len(true_annotations),
            'yolo_count': len(yolo_predictions),
            'frcnn_count': len(frcnn_predictions)
        }
    
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        traceback.print_exc()
        return None

def generate_summary_visualizations(results, output_dir):
    """生成汇总可视化图表"""
    if not results:
        print("没有结果可以生成汇总可视化")
        return
    
    # 提取指标
    image_names = []
    yolo_precision = []
    yolo_recall = []
    yolo_f1 = []
    yolo_ap = []
    yolo_time = []
    
    frcnn_precision = []
    frcnn_recall = []
    frcnn_f1 = []
    frcnn_ap = []
    frcnn_time = []
    
    true_counts = []
    yolo_counts = []
    frcnn_counts = []
    
    for result in results:
        image_names.append(result['image_name'])
        yolo_metrics = result['yolo_metrics']
        frcnn_metrics = result['frcnn_metrics']
        
        yolo_precision.append(yolo_metrics['precision'])
        yolo_recall.append(yolo_metrics['recall'])
        yolo_f1.append(yolo_metrics['f1_score'])
        yolo_ap.append(yolo_metrics['average_precision'])
        yolo_time.append(yolo_metrics['inference_time'])
        
        frcnn_precision.append(frcnn_metrics['precision'])
        frcnn_recall.append(frcnn_metrics['recall'])
        frcnn_f1.append(frcnn_metrics['f1_score'])
        frcnn_ap.append(frcnn_metrics['average_precision'])
        frcnn_time.append(frcnn_metrics['inference_time'])
        
        true_counts.append(result['true_count'])
        yolo_counts.append(result['yolo_count'])
        frcnn_counts.append(result['frcnn_count'])
    
    # 1. 绘制平均性能指标柱状图
    metrics = ['精确率', '召回率', 'F1分数', 'AP']
    yolo_means = [np.mean(yolo_precision), np.mean(yolo_recall), np.mean(yolo_f1), np.mean(yolo_ap)]
    frcnn_means = [np.mean(frcnn_precision), np.mean(frcnn_recall), np.mean(frcnn_f1), np.mean(frcnn_ap)]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, yolo_means, width, label='YOLOv8', color='tab:blue')
    plt.bar(x + width/2, frcnn_means, width, label='Faster R-CNN', color='tab:orange')
    
    for i, v in enumerate(yolo_means):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(frcnn_means):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.xticks(x, metrics)
    plt.ylabel('得分')
    plt.title('两种模型的平均性能指标对比')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(output_dir / "average_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制推理时间对比
    plt.figure(figsize=(10, 6))
    
    # 计算平均推理时间
    yolo_avg_time = np.mean(yolo_time)
    frcnn_avg_time = np.mean(frcnn_time)
    
    # 绘制条形图
    plt.bar(['YOLOv8', 'Faster R-CNN'], [yolo_avg_time, frcnn_avg_time], color=['tab:blue', 'tab:orange'])
    
    # 添加具体数值
    plt.text(0, yolo_avg_time + 0.01, f'{yolo_avg_time:.3f}s', ha='center')
    plt.text(1, frcnn_avg_time + 0.01, f'{frcnn_avg_time:.3f}s', ha='center')
    
    plt.ylabel('平均推理时间 (秒)')
    plt.title('两种模型的推理时间对比')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(output_dir / "inference_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 绘制散点图比较精确率和召回率
    plt.figure(figsize=(10, 8))
    
    plt.scatter(yolo_recall, yolo_precision, label='YOLOv8', alpha=0.7, s=50)
    plt.scatter(frcnn_recall, frcnn_precision, label='Faster R-CNN', alpha=0.7, s=50)
    
    # 添加平均点
    plt.scatter(np.mean(yolo_recall), np.mean(yolo_precision), marker='*', s=200, 
              color='blue', label='YOLOv8平均')
    plt.scatter(np.mean(frcnn_recall), np.mean(frcnn_precision), marker='*', s=200, 
              color='orange', label='Faster R-CNN平均')
    
    # 添加连接线
    for i in range(len(yolo_recall)):
        plt.plot([yolo_recall[i], frcnn_recall[i]], [yolo_precision[i], frcnn_precision[i]], 
               'k-', alpha=0.3)
    
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('两种模型的精确率-召回率散点图')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(str(output_dir / "pr_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 绘制对检测数量的对比
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(image_names))
    width = 0.25
    
    plt.bar(x - width, true_counts, width, label='真实标签', color='tab:green')
    plt.bar(x, yolo_counts, width, label='YOLOv8预测', color='tab:blue')
    plt.bar(x + width, frcnn_counts, width, label='Faster R-CNN预测', color='tab:orange')
    
    plt.xticks(x, image_names, rotation=45, ha='right')
    plt.ylabel('目标数量')
    plt.title('各图像中目标数量对比')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(output_dir / "object_counts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 绘制AP值箱线图
    plt.figure(figsize=(10, 6))
    
    data = [yolo_ap, frcnn_ap]
    plt.boxplot(data, labels=['YOLOv8', 'Faster R-CNN'], patch_artist=True,
              boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    
    plt.ylabel('平均精度 (AP)')
    plt.title('两种模型的AP值分布')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(output_dir / "ap_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 绘制性能雷达图
    plt.figure(figsize=(10, 10))
    
    # 准备数据
    categories = ['精确率', '召回率', 'F1分数', 'AP', '检出率', '推理速度']
    
    # 计算检出率
    yolo_detection_rate = np.mean(np.array(yolo_counts) / np.array(true_counts)) if np.any(true_counts) else 0
    frcnn_detection_rate = np.mean(np.array(frcnn_counts) / np.array(true_counts)) if np.any(true_counts) else 0
    
    # 归一化推理时间 (值越小越好，所以用倒数)
    max_time = max(np.max(yolo_time), np.max(frcnn_time))
    yolo_speed = 1 - (np.mean(yolo_time) / max_time)
    frcnn_speed = 1 - (np.mean(frcnn_time) / max_time)
    
    yolo_values = [np.mean(yolo_precision), np.mean(yolo_recall), np.mean(yolo_f1), 
                 np.mean(yolo_ap), yolo_detection_rate, yolo_speed]
    frcnn_values = [np.mean(frcnn_precision), np.mean(frcnn_recall), np.mean(frcnn_f1), 
                  np.mean(frcnn_ap), frcnn_detection_rate, frcnn_speed]
    
    # 创建雷达图
    ax = plt.subplot(111, polar=True)
    
    # 设置角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    # 添加数据
    yolo_values += yolo_values[:1]
    frcnn_values += frcnn_values[:1]
    
    # 绘制多边形和点
    ax.plot(angles, yolo_values, 'o-', linewidth=2, label='YOLOv8')
    ax.fill(angles, yolo_values, alpha=0.1)
    ax.plot(angles, frcnn_values, 'o-', linewidth=2, label='Faster R-CNN')
    ax.fill(angles, frcnn_values, alpha=0.1)
    
    # 设置标签
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    # 设置y轴范围
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
    
    plt.title('两种模型的性能雷达图')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(str(output_dir / "performance_radar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 创建综合评分卡
    plt.figure(figsize=(12, 8))
    
    # 定义评分指标和权重
    metrics = ['精确率', '召回率', 'F1分数', 'AP', '检出率', '推理速度']
    weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]  # 总和为1
    
    yolo_scores = np.array([np.mean(yolo_precision), np.mean(yolo_recall), np.mean(yolo_f1), 
                         np.mean(yolo_ap), yolo_detection_rate, yolo_speed])
    
    frcnn_scores = np.array([np.mean(frcnn_precision), np.mean(frcnn_recall), np.mean(frcnn_f1), 
                          np.mean(frcnn_ap), frcnn_detection_rate, frcnn_speed])
    
    # 计算加权分数
    yolo_weighted = yolo_scores * weights
    frcnn_weighted = frcnn_scores * weights
    
    yolo_total = np.sum(yolo_weighted) * 10  # 转换为0-10分制
    frcnn_total = np.sum(frcnn_weighted) * 10
    
    # 创建条形图
    x = np.arange(len(metrics))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制权重分数
    bars1 = ax.bar(x - width/2, yolo_weighted, width, label='YOLOv8', color='tab:blue')
    bars2 = ax.bar(x + width/2, frcnn_weighted, width, label='Faster R-CNN', color='tab:orange')
    
    # 添加标签
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('加权分数')
    ax.set_title('两种模型的性能评分卡')
    ax.legend()
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3点垂直偏移
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3点垂直偏移
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # 添加总分
    plt.figtext(0.15, 0.01, f'YOLOv8总分: {yolo_total:.2f}/10', fontsize=14, 
              bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.figtext(0.85, 0.01, f'Faster R-CNN总分: {frcnn_total:.2f}/10', fontsize=14, 
              bbox=dict(facecolor='orange', alpha=0.5))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 留出空间给总分
    
    plt.savefig(str(output_dir / "scorecard.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 生成各类型错误的对比
    plt.figure(figsize=(10, 8))
    
    # 汇总False positives和False negatives
    yolo_fp = [m['false_positives'] for m in [result['yolo_metrics'] for result in results]]
    yolo_fn = [m['false_negatives'] for m in [result['yolo_metrics'] for result in results]]
    
    frcnn_fp = [m['false_positives'] for m in [result['frcnn_metrics'] for result in results]]
    frcnn_fn = [m['false_negatives'] for m in [result['frcnn_metrics'] for result in results]]
    
    # 计算平均值
    yolo_avg_fp = np.mean(yolo_fp)
    yolo_avg_fn = np.mean(yolo_fn)
    frcnn_avg_fp = np.mean(frcnn_fp)
    frcnn_avg_fn = np.mean(frcnn_fn)
    
    # 绘制分组条形图
    x = np.arange(2)  # FP, FN
    width = 0.35
    
    plt.bar(x - width/2, [yolo_avg_fp, yolo_avg_fn], width, label='YOLOv8', color=['tab:red', 'tab:blue'])
    plt.bar(x + width/2, [frcnn_avg_fp, frcnn_avg_fn], width, label='Faster R-CNN', color=['tab:pink', 'tab:cyan'])
    
    # 添加标签
    plt.xticks(x, ['假阳性 (FP)', '假阴性 (FN)'])
    plt.ylabel('平均错误数量')
    plt.title('两种模型的错误类型对比')
    plt.legend()
    
    # 添加具体数值
    plt.text(x[0] - width/2, yolo_avg_fp + 0.1, f'{yolo_avg_fp:.1f}', ha='center')
    plt.text(x[1] - width/2, yolo_avg_fn + 0.1, f'{yolo_avg_fn:.1f}', ha='center')
    plt.text(x[0] + width/2, frcnn_avg_fp + 0.1, f'{frcnn_avg_fp:.1f}', ha='center')
    plt.text(x[1] + width/2, frcnn_avg_fn + 0.1, f'{frcnn_avg_fn:.1f}', ha='center')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(output_dir / "error_types.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. 生成HTML报告
    html_report = generate_html_report(results, output_dir)
    with open(output_dir / 'comparison_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)

def generate_html_report(results, output_dir):
    """生成HTML格式的比较报告"""
    # 计算平均指标
    yolo_precision = np.mean([r['yolo_metrics']['precision'] for r in results])
    yolo_recall = np.mean([r['yolo_metrics']['recall'] for r in results])
    yolo_f1 = np.mean([r['yolo_metrics']['f1_score'] for r in results])
    yolo_ap = np.mean([r['yolo_metrics']['average_precision'] for r in results])
    yolo_time = np.mean([r['yolo_metrics']['inference_time'] for r in results])
    
    frcnn_precision = np.mean([r['frcnn_metrics']['precision'] for r in results])
    frcnn_recall = np.mean([r['frcnn_metrics']['recall'] for r in results])
    frcnn_f1 = np.mean([r['frcnn_metrics']['f1_score'] for r in results])
    frcnn_ap = np.mean([r['frcnn_metrics']['average_precision'] for r in results])
    frcnn_time = np.mean([r['frcnn_metrics']['inference_time'] for r in results])
    
    # 生成报告
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>YOLOv8和Faster R-CNN模型比较报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .winner {{ background-color: #d4edda; font-weight: bold; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin-bottom: 20px; }}
            .chart {{ margin: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .header {{ background-color: #4a6572; color: white; padding: 20px; text-align: center; }}
            .conclusion {{ background-color: #f8f9fa; padding: 15px; border-left: 5px solid #5c6bc0; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>YOLOv8和Faster R-CNN模型比较报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>总体性能对比</h2>
            <p>本报告对比了YOLOv8和Faster R-CNN两种目标检测模型在相同测试集上的性能表现。测试包含 {len(results)} 张图像。</p>
        </div>
        
        <h2>平均性能指标</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>YOLOv8</th>
                <th>Faster R-CNN</th>
                <th>性能差异</th>
                <th>更优模型</th>
            </tr>
            <tr>
                <td>精确率 (Precision)</td>
                <td class="{get_winner_class(yolo_precision, frcnn_precision)}">{yolo_precision:.4f}</td>
                <td class="{get_winner_class(frcnn_precision, yolo_precision)}">{frcnn_precision:.4f}</td>
                <td>{abs(yolo_precision - frcnn_precision):.4f}</td>
                <td>{get_winner_name(yolo_precision, frcnn_precision)}</td>
            </tr>
            <tr>
                <td>召回率 (Recall)</td>
                <td class="{get_winner_class(yolo_recall, frcnn_recall)}">{yolo_recall:.4f}</td>
                <td class="{get_winner_class(frcnn_recall, yolo_recall)}">{frcnn_recall:.4f}</td>
                <td>{abs(yolo_recall - frcnn_recall):.4f}</td>
                <td>{get_winner_name(yolo_recall, frcnn_recall)}</td>
            </tr>
            <tr>
                <td>F1分数</td>
                <td class="{get_winner_class(yolo_f1, frcnn_f1)}">{yolo_f1:.4f}</td>
                <td class="{get_winner_class(frcnn_f1, yolo_f1)}">{frcnn_f1:.4f}</td>
                <td>{abs(yolo_f1 - frcnn_f1):.4f}</td>
                <td>{get_winner_name(yolo_f1, frcnn_f1)}</td>
            </tr>
            <tr>
                <td>平均精度 (AP)</td>
                <td class="{get_winner_class(yolo_ap, frcnn_ap)}">{yolo_ap:.4f}</td>
                <td class="{get_winner_class(frcnn_ap, yolo_ap)}">{frcnn_ap:.4f}</td>
                <td>{abs(yolo_ap - frcnn_ap):.4f}</td>
                <td>{get_winner_name(yolo_ap, frcnn_ap)}</td>
            </tr>
            <tr>
                <td>推理时间 (秒)</td>
                <td class="{get_winner_class(frcnn_time, yolo_time)}">{yolo_time:.4f}</td>
                <td class="{get_winner_class(yolo_time, frcnn_time)}">{frcnn_time:.4f}</td>
                <td>{abs(yolo_time - frcnn_time):.4f}</td>
                <td>{get_winner_name(frcnn_time, yolo_time)}</td>
            </tr>
        </table>
        
        <div class="conclusion">
            <h2>结论</h2>
            <p>
                根据以上性能对比，我们可以得出以下结论:
            </p>
            <ul>
                <li><strong>精度方面:</strong> {get_conclusion_text('精度', yolo_precision, yolo_recall, yolo_f1, yolo_ap, frcnn_precision, frcnn_recall, frcnn_f1, frcnn_ap)}</li>
                <li><strong>速度方面:</strong> {get_speed_conclusion(yolo_time, frcnn_time)}</li>
                <li><strong>综合评价:</strong> {get_overall_conclusion(yolo_precision, yolo_recall, yolo_f1, yolo_ap, yolo_time, frcnn_precision, frcnn_recall, frcnn_f1, frcnn_ap, frcnn_time)}</li>
            </ul>
        </div>
        
        <h2>可视化分析</h2>
        <div class="chart-container">
            <div class="chart">
                <img src="average_performance.png" alt="平均性能对比" width="500">
                <p>图1: 两种模型的平均性能指标对比</p>
            </div>
            <div class="chart">
                <img src="inference_time.png" alt="推理时间对比" width="500">
                <p>图2: 两种模型的平均推理时间对比</p>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart">
                <img src="pr_scatter.png" alt="PR散点图" width="500">
                <p>图3: 精确率-召回率散点图对比</p>
            </div>
            <div class="chart">
                <img src="performance_radar.png" alt="性能雷达图" width="500">
                <p>图4: 两种模型的性能雷达图</p>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart">
                <img src="ap_boxplot.png" alt="AP箱线图" width="500">
                <p>图5: AP值分布箱线图</p>
            </div>
            <div class="chart">
                <img src="error_types.png" alt="错误类型对比" width="500">
                <p>图6: 两种模型的错误类型对比</p>
            </div>
        </div>
        
        <h2>详细结果</h2>
        <table>
            <tr>
                <th>图像名称</th>
                <th>真实目标数</th>
                <th>YOLOv8检测数</th>
                <th>YOLOv8 F1</th>
                <th>YOLOv8 AP</th>
                <th>YOLOv8时间(秒)</th>
                <th>Faster R-CNN检测数</th>
                <th>Faster R-CNN F1</th>
                <th>Faster R-CNN AP</th>
                <th>Faster R-CNN时间(秒)</th>
            </tr>
    """
    
    # 添加每张图像的详细结果
    for result in results:
        yolo_metrics = result['yolo_metrics']
        frcnn_metrics = result['frcnn_metrics']
        
        html += f"""
            <tr>
                <td>{result['image_name']}</td>
                <td>{result['true_count']}</td>
                <td>{result['yolo_count']}</td>
                <td>{yolo_metrics['f1_score']:.3f}</td>
                <td>{yolo_metrics['average_precision']:.3f}</td>
                <td>{yolo_metrics['inference_time']:.3f}</td>
                <td>{result['frcnn_count']}</td>
                <td>{frcnn_metrics['f1_score']:.3f}</td>
                <td>{frcnn_metrics['average_precision']:.3f}</td>
                <td>{frcnn_metrics['inference_time']:.3f}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>样本检测对比</h2>
        <div class="chart-container">
    """
    
    # 添加样本对比图
    sample_images = sorted(list(Path(output_dir).glob("compare_*.png")))[:3]  # 最多显示3张
    for img_path in sample_images:
        html += f"""
            <div class="chart">
                <img src="{img_path.name}" alt="对比图" width="800">
                <p>图像对比: {img_path.stem.replace('compare_', '')}</p>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def get_winner_class(value1, value2):
    """根据两个值的比较返回CSS类"""
    if value1 > value2:
        return "winner"
    return ""

def get_winner_name(value1, value2, eps=1e-6):
    """根据两个值的比较返回获胜者名称"""
    if abs(value1 - value2) < eps:
        return "相似"
    elif value1 > value2:
        return "YOLOv8"
    else:
        return "Faster R-CNN"

def get_conclusion_text(aspect, yolo_precision, yolo_recall, yolo_f1, yolo_ap, 
                      frcnn_precision, frcnn_recall, frcnn_f1, frcnn_ap):
    """生成关于模型比较的结论文本"""
    yolo_avg = (yolo_precision + yolo_recall + yolo_f1 + yolo_ap) / 4
    frcnn_avg = (frcnn_precision + frcnn_recall + frcnn_f1 + frcnn_ap) / 4
    
    diff = abs(yolo_avg - frcnn_avg)
    
    if diff < 0.05:
        return f"两个模型在{aspect}方面表现相近，没有明显优势。YOLOv8的平均指标为{yolo_avg:.4f}，Faster R-CNN的平均指标为{frcnn_avg:.4f}。"
    elif yolo_avg > frcnn_avg:
        return f"YOLOv8在{aspect}方面表现更好，平均指标为{yolo_avg:.4f}，比Faster R-CNN高出{diff:.4f}。"
    else:
        return f"Faster R-CNN在{aspect}方面表现更好，平均指标为{frcnn_avg:.4f}，比YOLOv8高出{diff:.4f}。"

def get_speed_conclusion(yolo_time, frcnn_time):
    """生成关于速度的结论文本"""
    ratio = frcnn_time / yolo_time if yolo_time > 0 else float('inf')
    
    if ratio > 2:
        return f"YOLOv8在速度方面具有显著优势，平均推理时间为{yolo_time:.4f}秒，比Faster R-CNN快{ratio:.1f}倍。"
    elif ratio > 1.2:
        return f"YOLOv8在速度方面略有优势，平均推理时间为{yolo_time:.4f}秒，比Faster R-CNN快{ratio:.1f}倍。"
    elif ratio >= 0.8:
        return f"两个模型的推理速度相近，YOLOv8为{yolo_time:.4f}秒，Faster R-CNN为{frcnn_time:.4f}秒。"
    else:
        inverse_ratio = yolo_time / frcnn_time
        return f"Faster R-CNN在速度方面具有优势，平均推理时间为{frcnn_time:.4f}秒，比YOLOv8快{inverse_ratio:.1f}倍。"

def get_overall_conclusion(yolo_precision, yolo_recall, yolo_f1, yolo_ap, yolo_time,
                         frcnn_precision, frcnn_recall, frcnn_f1, frcnn_ap, frcnn_time):
    """生成综合评价结论"""
    # 计算平均精度指标
    yolo_accuracy = (yolo_precision + yolo_recall + yolo_f1 + yolo_ap) / 4
    frcnn_accuracy = (frcnn_precision + frcnn_recall + frcnn_f1 + frcnn_ap) / 4
    
    # 计算速度得分（越快越高分）
    max_time = max(yolo_time, frcnn_time)
    yolo_speed_score = 1 - (yolo_time / max_time) if max_time > 0 else 0.5
    frcnn_speed_score = 1 - (frcnn_time / max_time) if max_time > 0 else 0.5
    
    # 综合得分（精度占80%，速度占20%）
    yolo_overall = yolo_accuracy * 0.8 + yolo_speed_score * 0.2
    frcnn_overall = frcnn_accuracy * 0.8 + frcnn_speed_score * 0.2
    
    if abs(yolo_overall - frcnn_overall) < 0.05:
        return "综合考虑精度和速度，两个模型的整体表现相当，可以根据具体应用场景选择适合的模型。"
    elif yolo_overall > frcnn_overall:
        return f"综合考虑精度和速度，YOLOv8的整体表现更优，综合得分为{yolo_overall:.4f}，适合需要兼顾精度和速度的场景，特别是实时检测应用。"
    else:
        return f"综合考虑精度和速度，Faster R-CNN的整体表现更优，综合得分为{frcnn_overall:.4f}，适合对检测精度要求较高的场景。"

def main():
    parser = argparse.ArgumentParser(description='对比YOLOv8和Faster R-CNN两种目标检测模型')
    parser.add_argument('--yolo_model', type=str, required=False, help='YOLOv8模型路径')
    parser.add_argument('--frcnn_model', type=str, required=False, help='Faster R-CNN模型路径')
    parser.add_argument('--data_dir', type=str, required=False, help='测试数据目录')
    parser.add_argument('--config_path', type=str, required=False, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, required=False, help='输出目录')
    parser.add_argument('--samples', type=int, default=5, help='随机抽取的样本数量')
    parser.add_argument('--yolo_conf', type=float, default=0.25, help='YOLOv8置信度阈值')
    parser.add_argument('--frcnn_conf', type=float, default=0.5, help='Faster R-CNN置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置基础目录
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # 设置YOLO和Faster R-CNN目录
    yolo_dir = base_dir / "yolo_code"
    frcnn_dir = base_dir / "frcnn_code"
    
    # 设置配置文件路径
    config_path = args.config_path if args.config_path else yolo_dir / "config" / "insects.yaml"
    
    # 设置输出目录
    results_dir = frcnn_dir / "results"
    comparisons_base_dir = results_dir / "comparisons"
    comparisons_base_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir if args.output_dir else comparisons_base_dir / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 设置日志
    log_dir = results_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir)
    logger.info(f"比较结果将保存到: {output_dir}")
    
    # 尝试加载字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        class_names = config['names']
        logger.info(f"加载类别信息: {class_names}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 确定模型路径
    yolo_model_path = args.yolo_model if args.yolo_model else yolo_dir / "results" / "train" / "weights" / "best.pt"
    frcnn_model_path = args.frcnn_model if args.frcnn_model else frcnn_dir / "results" / "models" / "best_model.pt"
    
    # 如果指定路径不存在，尝试查找其他模型
    if not Path(yolo_model_path).exists():
        # 尝试查找最新的YOLOv8模型
        train_dirs = sorted(list((yolo_dir / "results").glob("train*")))
        if train_dirs:
            for train_dir in reversed(train_dirs):
                best_model = train_dir / "weights" / "best.pt"
                if best_model.exists():
                    yolo_model_path = best_model
                    logger.info(f"找到YOLOv8模型: {yolo_model_path}")
                    break
    
    if not Path(frcnn_model_path).exists():
        # 尝试查找最新的Faster R-CNN模型
        model_dirs = sorted(list((frcnn_dir / "results" / "models").glob("*.pt")))
        if model_dirs:
            frcnn_model_path = model_dirs[-1]
            logger.info(f"找到Faster R-CNN模型: {frcnn_model_path}")
    
    logger.info(f"使用YOLOv8模型: {yolo_model_path}")
    logger.info(f"使用Faster R-CNN模型: {frcnn_model_path}")
    
    # 加载模型
    yolo_model, frcnn_model = load_models(yolo_model_path, frcnn_model_path, len(class_names), device)
    
    if not yolo_model or not frcnn_model:
        logger.error("模型加载失败，请检查路径是否正确")
        return
    
    # 确定数据目录
    data_dir = args.data_dir if args.data_dir else base_dir / "data" / "insects" / "val"
    data_dir = Path(data_dir)
    
    # 收集图像和标签
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    
    if not images_dir.exists() or not annotations_dir.exists():
        logger.error(f"数据目录结构不正确: {data_dir}")
        return
    
    # 收集所有图像和对应的XML标注
    image_files = []
    xml_files = []
    
    for img_ext in ['.jpg', '.jpeg', '.png']:
        for img_file in images_dir.glob(f'*{img_ext}'):
            base_name = img_file.stem
            xml_file = annotations_dir / f"{base_name}.xml"
            
            if xml_file.exists():
                image_files.append(img_file)
                xml_files.append(xml_file)
    
    if not image_files:
        logger.error(f"在 {images_dir} 中未找到带有对应XML标注的图像文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个带有XML标注的图像文件")
    
    # 随机抽样
    if args.samples < len(image_files):
        indices = random.sample(range(len(image_files)), args.samples)
        image_files = [image_files[i] for i in indices]
        xml_files = [xml_files[i] for i in indices]
        logger.info(f"随机抽取 {args.samples} 个样本进行比较")
    
    # 创建结果列表
    results = []
    
    # 比较模型
    for i, (img_path, xml_path) in enumerate(zip(image_files, xml_files)):
        logger.info(f"处理图像 {i+1}/{len(image_files)}: {img_path.name}")
        
        result = compare_predictions(
            img_path, 
            xml_path, 
            yolo_model, 
            frcnn_model, 
            class_names, 
            device, 
            output_dir,
            yolo_conf=args.yolo_conf,
            frcnn_conf=args.frcnn_conf,
            iou_threshold=args.iou_threshold
        )
        
        if result:
            results.append(result)
    
    # 生成汇总可视化
    if results:
        logger.info("生成汇总可视化...")
        generate_summary_visualizations(results, output_dir)
        logger.info(f"比较完成，结果已保存到 {output_dir}")
    else:
        logger.error("没有成功处理任何图像，无法生成汇总可视化")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()