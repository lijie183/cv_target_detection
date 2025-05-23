import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
import matplotlib
import yaml
import shutil
import logging
from datetime import datetime

# 设置中文字体
matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志器"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # 添加控制台handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    
    return logger

def plot_images_with_boxes(images, targets, output_path, figsize=(12, 12), max_imgs=16, class_names=None):
    """绘制带有边界框的图像"""
    num_imgs = min(len(images), max_imgs)
    fig = plt.figure(figsize=figsize)
    
    for i in range(num_imgs):
        ax = plt.subplot(int(np.ceil(num_imgs / 4)), 4, i + 1)
        
        # 转换图像并显示
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        
        # 绘制边界框
        if targets is not None:
            # 筛选出对应图像的边界框
            bbox_idx = targets[:, 0] == i
            bboxes = targets[bbox_idx, 1:]
            
            for box in bboxes:
                cls_id = int(box[0])
                x, y, w, h = box[1:].tolist()
                
                # YOLO格式转换为像素坐标
                h_img, w_img = img.shape[:2]
                x1 = (x - w/2) * w_img
                y1 = (y - h/2) * h_img
                x2 = (x + w/2) * w_img
                y2 = (y + h/2) * h_img
                
                # 绘制边界框
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                    edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                # 添加类别标签
                if class_names is not None:
                    plt.text(x1, y1 - 5, class_names[cls_id], 
                            color='white', fontsize=8,
                            bbox=dict(facecolor='r', alpha=0.5))
        
        ax.set_title(f'Image {i}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def load_yaml_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_iou(box1, box2):
    """计算IoU"""
    # 从YOLO格式转换为边界框坐标
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    
    # 计算交集区域
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    # 计算交集面积
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # 计算并集面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / union_area

def xywh2xyxy(x):
    """将YOLO格式的边界框转换为(x1, y1, x2, y2)格式"""
    y = torch.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y