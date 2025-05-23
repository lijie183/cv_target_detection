import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from matplotlib.font_manager import FontProperties
import torchvision.transforms as T

from config import *

def visualize_batch(image, target, img_id, epoch, is_validation=False):
    """可视化图像和目标框"""
    # 转换图像为numpy数组用于matplotlib显示
    img_np = image.permute(1, 2, 0).numpy()
    
    # 对图像进行标准化还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    # 创建图像
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # 显示图像
    ax.imshow(img_np)
    
    # 获取图像尺寸
    height, width = img_np.shape[:2]
    
    # 在图像上绘制边界框
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            # 将YOLO格式的框 (x_center, y_center, width, height) 转换为 (x_min, y_min, width, height)
            x_center, y_center, box_width, box_height = box.numpy()
            
            # 将归一化坐标转换为像素坐标
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            
            # 计算左上角坐标
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            
            # 获取类别颜色
            color = COLOR_PALETTE[int(label) % len(COLOR_PALETTE)]
            color = [c/255 for c in color]  # 转换为matplotlib需要的0-1范围
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x_min, y_min), box_width, box_height, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加类别标签
            class_name = VOC_CLASSES[int(label)]
            ax.text(
                x_min, y_min - 5, class_name, 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # 添加图像ID和epoch信息
    title = f"{'Validation' if is_validation else 'Training'} - Epoch {epoch+1} - {img_id}"
    ax.set_title(title)
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 设置紧凑布局
    plt.tight_layout()
    
    return fig

def visualize_predictions(image, predictions, target, img_id, class_names, conf_threshold=0.5):
    """可视化检测结果"""
    # 转换图像为numpy数组用于matplotlib显示
    img_np = image.permute(1, 2, 0).numpy()
    
    # 对图像进行标准化还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 显示图像和真实标注
    ax1.imshow(img_np)
    ax1.set_title(f"Ground Truth - {img_id}")
    
    # 获取图像尺寸
    height, width = img_np.shape[:2]
    
    # 在图像上绘制真实边界框
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            # 将YOLO格式的框 (x_center, y_center, width, height) 转换为 (x_min, y_min, width, height)
            x_center, y_center, box_width, box_height = box.numpy()
            
            # 将归一化坐标转换为像素坐标
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            
            # 计算左上角坐标
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            
            # 获取类别颜色
            color = COLOR_PALETTE[int(label) % len(COLOR_PALETTE)]
            color = [c/255 for c in color]  # 转换为matplotlib需要的0-1范围
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x_min, y_min), box_width, box_height, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax1.add_patch(rect)
            
            # 添加类别标签
            class_name = class_names[int(label)]
            ax1.text(
                x_min, y_min - 5, class_name, 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # 显示图像和预测结果
    ax2.imshow(img_np)
    ax2.set_title(f"Predictions - {img_id}")
    
    # 在图像上绘制预测边界框
    if predictions is not None and len(predictions) > 0:
        for i in range(len(predictions)):
            # 获取预测框和置信度
            box = predictions[i, :4].cpu().numpy()
            conf = predictions[i, 4].cpu().numpy()
            
            if conf < conf_threshold:
                continue
            
            # 获取类别分数和预测的类别
            cls_scores = predictions[i, 5:].cpu().numpy()
            cls_id = np.argmax(cls_scores)
            
            # 将YOLO格式的框转换为矩形框
            x1, y1, x2, y2 = box
            
            # 获取类别颜色
            color = COLOR_PALETTE[int(cls_id) % len(COLOR_PALETTE)]
            color = [c/255 for c in color]  # 转换为matplotlib需要的0-1范围
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax2.add_patch(rect)
            
            # 添加类别标签和置信度
            class_name = class_names[int(cls_id)]
            ax2.text(
                x1, y1 - 5, f"{class_name} {conf:.2f}", 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # 隐藏坐标轴
    ax1.axis('off')
    ax2.axis('off')
    
    # 设置紧凑布局
    plt.tight_layout()
    
    return fig

def visualize_test_predictions(image, detections, color_palette=None):
    """可视化测试图像上的检测结果"""
    if color_palette is None:
        color_palette = COLOR_PALETTE
    
    # 创建图像副本
    img_vis = image.copy()
    
    # 加载中文字体
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
    except:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
    
    # 在图像上绘制检测框
    for det in detections:
        # 获取框、类别和置信度
        box = det['bbox']  # [x1, y1, x2, y2]
        category_id = det['category_id']
        category_name = det['category_name']
        score = det['score']
        
        # 获取类别颜色
        color = color_palette[category_id % len(color_palette)]
        
        # 绘制边界框
        x1, y1, x2, y2 = box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
        
        # 添加类别标签和置信度
        label = f"{category_name} {score:.2f}"
        
        # 获取文本框大小
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制文本背景
        cv2.rectangle(
            img_vis, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # 绘制文本
        cv2.putText(
            img_vis, 
            label, 
            (x1, y1 - 5), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness, 
            cv2.LINE_AA
        )
    
    return img_vis