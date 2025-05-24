# 目标检测实验方案设计

根据您提供的数据集信息和要求，我将设计一套基于YOLOv10的昆虫目标检测实验方案。这套方案将包含先进的技术和创新点，确保实验结果具有高精度和实用性。

## 实验流程概述

1. **数据准备与预处理**：解析XML格式的标注文件，转换为YOLO格式
2. **数据增强与分析**：实现多种数据增强策略，分析并可视化数据分布
3. **模型构建与优化**：基于YOLOv10架构，引入多项创新改进
4. **训练与监控**：实现训练过程监控、早停机制和丰富的可视化
5. **评估与测试**：多指标评估模型性能，可视化检测结果

## 创新点概述

1. **FocalAttention模块**：引入注意力机制，增强对小目标昆虫的感知能力
2. **自适应特征融合**：根据不同尺度的昆虫特征动态调整融合权重
3. **标签平滑与混合损失函数**：结合Focal Loss和IoU Loss提高检测准确性
4. **多尺度训练与推理**：适应昆虫体型差异大的特点
5. **梯度集中学习策略**：针对昆虫特征学习的优化策略

## 文件结构设计

```
yolo_code/
├── scripts/
│   ├── data_preprocessing.py  # 数据预处理和转换
│   ├── data_augmentation.py   # 数据增强和可视化
│   ├── model.py               # 改进的YOLOv10模型定义
│   ├── train.py               # 模型训练与监控
│   └── evaluate.py            # 模型评估与结果可视化
├── results/
│   ├── logs/                  # 训练日志
│   ├── models/                # 保存的模型
│   ├── visualizations/        # 数据和结果可视化
│   ├── metrics/               # 评估指标
│   └── predictions/           # 模型预测结果
```

## 代码实现

下面是五个主要脚本的完整实现：

### 1. 数据预处理脚本

````python
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import shutil
from tqdm import tqdm
import logging
import json
from sklearn.model_selection import train_test_split
import cv2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('..', 'results', 'logs', 'data_preprocessing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保结果目录存在
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 创建必要的文件夹
def create_directories():
    directories = [
        os.path.join('..', 'results'),
        os.path.join('..', 'results', 'logs'),
        os.path.join('..', 'results', 'models'),
        os.path.join('..', 'results', 'visualizations'),
        os.path.join('..', 'results', 'visualizations', 'dataset'),
        os.path.join('..', 'results', 'metrics'),
        os.path.join('..', 'results', 'predictions'),
        os.path.join('..', 'data', 'yolo_format')
    ]
    
    for directory in directories:
        ensure_dir(directory)
    
    logger.info("创建目录结构完成")

# 解析XML文件
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像信息
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 获取所有目标信息
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'width': width,
        'height': height,
        'objects': objects
    }

# 将边界框转换为YOLO格式 (x_center, y_center, width, height) - 归一化坐标
def convert_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 归一化
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

# 转换数据集到YOLO格式
def convert_dataset_to_yolo(data_dir, yolo_dir, class_mapping):
    subsets = ['train', 'val', 'test']
    
    # 准备YOLO格式目录
    for subset in subsets:
        ensure_dir(os.path.join(yolo_dir, subset, 'images'))
        ensure_dir(os.path.join(yolo_dir, subset, 'labels'))
    
    class_stats = {subset: {} for subset in subsets}
    image_sizes = {subset: [] for subset in subsets}
    
    # 处理每个子集
    for subset in subsets:
        logger.info(f"处理{subset}数据集...")
        
        anno_dir = os.path.join(data_dir, 'insects', subset, 'annotations')
        img_dir = os.path.join(data_dir, 'insects', subset, 'images')
        
        xml_files = [f for f in os.listdir(anno_dir) if f.endswith('.xml')]
        
        for xml_file in tqdm(xml_files, desc=f"转换{subset}集"):
            # 解析XML
            xml_path = os.path.join(anno_dir, xml_file)
            try:
                data = parse_xml(xml_path)
            except Exception as e:
                logger.error(f"解析XML文件{xml_file}时出错: {e}")
                continue
            
            # 准备图像文件
            img_filename = os.path.splitext(xml_file)[0] + '.jpeg'
            img_src = os.path.join(img_dir, img_filename)
            
            # 检查图像文件是否存在
            if not os.path.exists(img_src):
                logger.warning(f"图像文件不存在: {img_src}")
                continue
            
            # 复制图像文件
            img_dst = os.path.join(yolo_dir, subset, 'images', img_filename)
            shutil.copy(img_src, img_dst)
            
            # 创建标签文件
            label_filename = os.path.splitext(xml_file)[0] + '.txt'
            label_path = os.path.join(yolo_dir, subset, 'labels', label_filename)
            
            # 获取图像尺寸
            img_width = data['width']
            img_height = data['height']
            image_sizes[subset].append((img_width, img_height))
            
            # 写入YOLO格式标签
            with open(label_path, 'w') as f:
                for obj in data['objects']:
                    class_name = obj['name']
                    if class_name not in class_mapping:
                        logger.warning(f"未知类别: {class_name}，跳过该目标")
                        continue
                    
                    class_id = class_mapping[class_name]
                    
                    # 统计类别数量
                    if class_name not in class_stats[subset]:
                        class_stats[subset][class_name] = 0
                    class_stats[subset][class_name] += 1
                    
                    # 转换边界框并写入
                    yolo_bbox = convert_to_yolo_format(obj['bbox'], img_width, img_height)
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
    
    return class_stats, image_sizes

# 生成YOLO配置文件
def generate_yolo_config(yolo_dir, class_names):
    # 创建数据配置文件
    data_config = {
        'train': os.path.join(yolo_dir, 'train', 'images'),
        'val': os.path.join(yolo_dir, 'val', 'images'),
        'test': os.path.join(yolo_dir, 'test', 'images'),
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(yolo_dir, 'data.yaml'), 'w') as f:
        for key, value in data_config.items():
            if key == 'names':
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    logger.info(f"生成YOLO配置文件: {os.path.join(yolo_dir, 'data.yaml')}")

# 可视化数据集统计
def visualize_dataset_stats(class_stats, image_sizes, output_dir):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
    
    # 1. 类别分布可视化
    plt.figure(figsize=(12, 8))
    
    for i, subset in enumerate(['train', 'val', 'test']):
        plt.subplot(1, 3, i+1)
        if not class_stats[subset]:
            plt.text(0.5, 0.5, f"无{subset}数据", 
                     horizontalalignment='center', verticalalignment='center')
            continue
            
        labels = list(class_stats[subset].keys())
        sizes = list(class_stats[subset].values())
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title(f"{subset}集类别分布", fontproperties=font)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    
    # 2. 各子集样本数量
    plt.figure(figsize=(10, 6))
    subsets = ['train', 'val', 'test']
    sample_counts = [sum(class_stats[subset].values()) for subset in subsets]
    
    plt.bar(subsets, sample_counts, color=['blue', 'green', 'orange'])
    for i, count in enumerate(sample_counts):
        plt.text(i, count + 10, str(count), ha='center', fontproperties=font)
    
    plt.title('各子集样本数量统计', fontproperties=font)
    plt.ylabel('样本数量', fontproperties=font)
    plt.savefig(os.path.join(output_dir, 'sample_counts.png'), dpi=300)
    
    # 3. 各类别在各子集中的分布
    plt.figure(figsize=(14, 8))
    
    all_classes = set()
    for subset in subsets:
        all_classes.update(class_stats[subset].keys())
    
    all_classes = sorted(list(all_classes))
    subset_colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bar_width = 0.25
    index = np.arange(len(all_classes))
    
    for i, subset in enumerate(subsets):
        counts = [class_stats[subset].get(cls, 0) for cls in all_classes]
        plt.bar(index + i*bar_width, counts, bar_width, 
                label=subset, color=subset_colors[i])
    
    plt.xlabel('类别', fontproperties=font)
    plt.ylabel('数量', fontproperties=font)
    plt.title('各类别在不同子集中的分布', fontproperties=font)
    plt.xticks(index + bar_width, all_classes, rotation=45, fontproperties=font)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution_by_subset.png'), dpi=300)
    
    # 4. 图像尺寸分布
    plt.figure(figsize=(15, 5))
    
    for i, subset in enumerate(subsets):
        plt.subplot(1, 3, i+1)
        if not image_sizes[subset]:
            plt.text(0.5, 0.5, f"无{subset}数据", 
                     horizontalalignment='center', verticalalignment='center')
            continue
            
        widths = [size[0] for size in image_sizes[subset]]
        heights = [size[1] for size in image_sizes[subset]]
        
        plt.scatter(widths, heights, alpha=0.5)
        plt.title(f"{subset}集图像尺寸分布", fontproperties=font)
        plt.xlabel('宽度', fontproperties=font)
        plt.ylabel('高度', fontproperties=font)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_size_distribution.png'), dpi=300)
    
    logger.info(f"数据集统计可视化已保存到: {output_dir}")

# 分析对象尺寸分布
def analyze_object_sizes(data_dir):
    subsets = ['train', 'val', 'test']
    object_sizes = {subset: [] for subset in subsets}
    
    for subset in subsets:
        anno_dir = os.path.join(data_dir, 'insects', subset, 'annotations')
        xml_files = [f for f in os.listdir(anno_dir) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(anno_dir, xml_file)
            try:
                data = parse_xml(xml_path)
                for obj in data['objects']:
                    bbox = obj['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    object_sizes[subset].append({
                        'class': obj['name'],
                        'width': width,
                        'height': height,
                        'area': area,
                        'aspect_ratio': width / height
                    })
            except Exception as e:
                logger.error(f"分析{xml_file}目标尺寸时出错: {e}")
    
    return object_sizes

# 可视化对象尺寸分布
def visualize_object_sizes(object_sizes, output_dir):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
    
    for subset in object_sizes:
        if not object_sizes[subset]:
            continue
            
        df = pd.DataFrame(object_sizes[subset])
        
        # 1. 目标面积分布
        plt.figure(figsize=(10, 6))
        plt.hist(df['area'], bins=30)
        plt.title(f"{subset}集目标面积分布", fontproperties=font)
        plt.xlabel('面积（像素）', fontproperties=font)
        plt.ylabel('频率', fontproperties=font)
        plt.savefig(os.path.join(output_dir, f'{subset}_object_area_distribution.png'), dpi=300)
        
        # 2. 宽高比分布
        plt.figure(figsize=(10, 6))
        plt.hist(df['aspect_ratio'], bins=30)
        plt.title(f"{subset}集目标宽高比分布", fontproperties=font)
        plt.xlabel('宽高比', fontproperties=font)
        plt.ylabel('频率', fontproperties=font)
        plt.savefig(os.path.join(output_dir, f'{subset}_object_aspect_ratio.png'), dpi=300)
        
        # 3. 按类别划分的目标尺寸散点图
        plt.figure(figsize=(12, 8))
        classes = df['class'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        
        for i, cls in enumerate(classes):
            mask = df['class'] == cls
            plt.scatter(df.loc[mask, 'width'], df.loc[mask, 'height'], 
                       label=cls, color=colors[i], alpha=0.7)
        
        plt.title(f"{subset}集不同类别目标尺寸分布", fontproperties=font)
        plt.xlabel('宽度（像素）', fontproperties=font)
        plt.ylabel('高度（像素）', fontproperties=font)
        plt.legend(prop=font)
        plt.savefig(os.path.join(output_dir, f'{subset}_object_size_by_class.png'), dpi=300)
    
    logger.info(f"目标尺寸分析可视化已保存到: {output_dir}")

# 可视化一些样本图像和标注框
def visualize_samples(data_dir, yolo_dir, class_names, output_dir, num_samples=5):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
    subsets = ['train', 'val', 'test']
    
    class_to_id = {name: i for i, name in enumerate(class_names)}
    id_to_class = {i: name for i, name in enumerate(class_names)}
    
    for subset in subsets:
        img_dir = os.path.join(data_dir, 'insects', subset, 'images')
        anno_dir = os.path.join(data_dir, 'insects', subset, 'annotations')
        
        # 获取所有有效的图像文件
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not img_files:
            continue
        
        # 随机选择样本
        samples = np.random.choice(img_files, min(num_samples, len(img_files)), replace=False)
        
        for i, img_file in enumerate(samples):
            img_path = os.path.join(img_dir, img_file)
            xml_file = os.path.splitext(img_file)[0] + '.xml'
            xml_path = os.path.join(anno_dir, xml_file)
            
            if not os.path.exists(xml_path):
                continue
            
            # 读取图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 解析XML
            data = parse_xml(xml_path)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            
            # 绘制边界框
            for obj in data['objects']:
                if obj['name'] not in class_to_id:
                    continue
                    
                xmin, ymin, xmax, ymax = obj['bbox']
                
                # 创建矩形
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                
                # 添加标签
                plt.text(xmin, ymin - 5, obj['name'], color='r', fontproperties=font,
                         bbox=dict(facecolor='white', alpha=0.7))
            
            plt.title(f"{subset}集样本 - {img_file}", fontproperties=font)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{subset}_sample_{i+1}.png'), dpi=300)
            plt.close()
    
    logger.info(f"样本可视化已保存到: {output_dir}")

def main():
    # 设置数据路径
    data_dir = os.path.join('..', '..', 'data')
    yolo_dir = os.path.join('..', 'data', 'yolo_format')
    output_dir = os.path.join('..', 'results', 'visualizations', 'dataset')
    
    # 创建必要的目录
    create_directories()
    
    # 获取所有类别
    class_names = set()
    for subset in ['train', 'val', 'test']:
        anno_dir = os.path.join(data_dir, 'insects', subset, 'annotations')
        
        if not os.path.exists(anno_dir):
            logger.warning(f"注释目录不存在: {anno_dir}")
            continue
            
        xml_files = [f for f in os.listdir(anno_dir) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(anno_dir, xml_file)
            try:
                data = parse_xml(xml_path)
                for obj in data['objects']:
                    class_names.add(obj['name'])
            except Exception as e:
                logger.error(f"解析XML文件{xml_file}时出错: {e}")
    
    class_names = sorted(list(class_names))
    class_mapping = {name: i for i, name in enumerate(class_names)}
    
    logger.info(f"检测到 {len(class_names)} 个类别: {class_names}")
    
    # 转换数据集
    class_stats, image_sizes = convert_dataset_to_yolo(data_dir, yolo_dir, class_mapping)
    
    # 生成YOLO配置文件
    generate_yolo_config(yolo_dir, class_names)
    
    # 可视化数据集统计
    visualize_dataset_stats(class_stats, image_sizes, output_dir)
    
    # 分析并可视化对象尺寸
    object_sizes = analyze_object_sizes(data_dir)
    visualize_object_sizes(object_sizes, output_dir)
    
    # 可视化样本
    visualize_samples(data_dir, yolo_dir, class_names, output_dir)
    
    logger.info("数据预处理完成")

if __name__ == "__main__":
    main()
````

### 2. 数据增强脚本

````python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import random
import shutil
from tqdm import tqdm
import logging
import torch
from PIL import Image, ImageEnhance, ImageOps
import albumentations as A
import yaml
from pathlib import Path
import matplotlib.patches as patches

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('..', 'results', 'logs', 'data_augmentation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保目录存在
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 读取YOLO格式数据配置
def read_data_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 读取YOLO格式标签
def read_yolo_label(label_path):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    boxes.append((cls_id, x_center, y_center, width, height))
    return boxes

# 将YOLO格式标签转回像素坐标
def yolo_to_pixel(box, img_width, img_height):
    cls_id, x_center, y_center, width, height = box
    
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    
    return cls_id, (xmin, ymin, xmax, ymax)

# 将像素坐标转回YOLO格式
def pixel_to_yolo(cls_id, bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # 归一化
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return (cls_id, x_center, y_center, width, height)

# 应用Albumentations数据增强
def apply_augmentation(image, boxes, aug_pipeline):
    # 将YOLO格式转换为Albumentations格式
    height, width = image.shape[:2]
    albu_boxes = []
    class_ids = []
    
    for box in boxes:
        cls_id, bbox = yolo_to_pixel(box, width, height)
        xmin, ymin, xmax, ymax = bbox
        albu_boxes.append([xmin, ymin, xmax, ymax])
        class_ids.append(cls_id)
    
    # 应用增强
    transformed = aug_pipeline(image=image, bboxes=albu_boxes, class_ids=class_ids)
    
    # 转回YOLO格式
    transformed_image = transformed['image']
    transformed_boxes = []
    
    for idx, (box, cls_id) in enumerate(zip(transformed['bboxes'], transformed['class_ids'])):
        xmin, ymin, xmax, ymax = box
        transformed_box = pixel_to_yolo(cls_id, (xmin, ymin, xmax, ymax), 
                                         transformed_image.shape[1], transformed_image.shape[0])
        transformed_boxes.append(transformed_box)
    
    return transformed_image, transformed_boxes

# 创建高级数据增强流水线
def create_augmentation_pipelines():
    # 1. 光照和颜色增强
    color_aug = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ], p=1.0),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))
    
    # 2. 几何变换
    geometry_aug = A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.7),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.15, 0.15), rotate=(-15, 15), p=0.7),
        ], p=1.0),
        A.OneOf([
            A.RandomCrop(height=900, width=900, p=0.3),
            A.RandomResizedCrop(height=1024, width=1024, scale=(0.8, 1.0), p=0.3),
        ], p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))
    
    # 3. 混合增强
    mixed_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
        ], p=0.3),
        A.RandomShadow(p=0.2),
        A.RandomFog(p=0.1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))
    
    # 4. 目标检测特定增强
    detection_aug = A.Compose([
        A.OneOf([
            A.RandomSizedBBoxSafeCrop(height=900, width=900, p=0.7),
            A.RandomResizedCrop(height=1024, width=1024, scale=(0.8, 1.0), p=0.3),
        ], p=0.8),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ToGray(p=0.1),
        A.GaussNoise(var_limit=(10, 30), p=0.2),
        A.Blur(blur_limit=3, p=0.1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))
    
    # 5. 小目标增强专用
    small_object_aug = A.Compose([
        A.OneOf([
            A.RandomScale(scale_limit=(0.1, 0.3), p=0.7),
            A.RandomSizedBBoxSafeCrop(height=800, width=800, p=0.7),
        ], p=0.8),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, p=0.7),
        A.CLAHE(clip_limit=3.0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))
    
    return {
        'color': color_aug,
        'geometry': geometry_aug,
        'mixed': mixed_aug,
        'detection': detection_aug,
        'small_object': small_object_aug
    }

# 应用数据增强并创建增强后的数据集
def augment_dataset(data_dir, output_dir, class_names, aug_factor=2):
    # 读取原始训练集
    train_images_dir = os.path.join(data_dir, 'train', 'images')
    train_labels_dir = os.path.join(data_dir, 'train', 'labels')
    
    # 创建输出目录
    augmented_images_dir = os.path.join(output_dir, 'train', 'images')
    augmented_labels_dir = os.path.join(output_dir, 'train', 'labels')
    ensure_dir(augmented_images_dir)
    ensure_dir(augmented_labels_dir)
    
    # 复制验证集和测试集
    for subset in ['val', 'test']:
        src_images_dir = os.path.join(data_dir, subset, 'images')
        src_labels_dir = os.path.join(data_dir, subset, 'labels')
        dst_images_dir = os.path.join(output_dir, subset, 'images')
        dst_labels_dir = os.path.join(output_dir, subset, 'labels')
        
        ensure_dir(dst_images_dir)
        ensure_dir(dst_labels_dir)
        
        # 复制图像和标签
        for filename in os.listdir(src_images_dir):
            shutil.copy(os.path.join(src_images_dir, filename), 
                        os.path.join(dst_images_dir, filename))
        
        for filename in os.listdir(src_labels_dir):
            shutil.copy(os.path.join(src_labels_dir, filename), 
                        os.path.join(dst_labels_dir, filename))
    
    # 创建增强流水线
    aug_pipelines = create_augmentation_pipelines()
    
    # 获取所有原始训练图像
    train_images = [f for f in os.listdir(train_images_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 首先复制原始图像和标签
    for filename in train_images:
        # 复制原始图像
        shutil.copy(os.path.join(train_images_dir, filename), 
                    os.path.join(augmented_images_dir, filename))
        
        # 复制原始标签
        label_filename = os.path.splitext(filename)[0] + '.txt'
        if os.path.exists(os.path.join(train_labels_dir, label_filename)):
            shutil.copy(os.path.join(train_labels_dir, label_filename), 
                        os.path.join(augmented_labels_dir, label_filename))
    
    # 分析每个类别的实例数量
    class_counts = {i: 0 for i in range(len(class_names))}
    for filename in train_images:
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_filename)
        if os.path.exists(label_path):
            boxes = read_yolo_label(label_path)
            for box in boxes:
                cls_id = box[0]
                class_counts[cls_id] += 1
    
    # 计算类别平衡权重
    max_count = max(class_counts.values())
    class_weights = {cls_id: max(1.0, min(3.0, max_count / count)) if count > 0 else 1.0 
                     for cls_id, count in class_counts.items()}
    
    logger.info(f"类别权重: {class_weights}")
    
    # 增强过程中的统计信息
    aug_stats = {
        'original': len(train_images),
        'augmented': 0,
        'by_class': {cls_id: 0 for cls_id in range(len(class_names))},
        'by_technique': {name: 0 for name in aug_pipelines}
    }
    
    # 对每张图像进行增强
    for filename in tqdm(train_images, desc="增强训练集"):
        # 读取图像和标签
        img_path = os.path.join(train_images_dir, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_filename)
        
        if not os.path.exists(label_path):
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"无法读取图像: {img_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = read_yolo_label(label_path)
        
        if not boxes:
            continue
        
        # 确定要应用的增强数量（基于类别平衡）
        classes_in_image = set(box[0] for box in boxes)
        max_weight = max(class_weights[cls_id] for cls_id in classes_in_image)
        num_augs = max(1, min(aug_factor, int(max_weight * 2)))
        
        # 应用增强
        for i in range(num_augs):
            # 选择增强类型
            aug_type = random.choice(list(aug_pipelines.keys()))
            aug_pipeline = aug_pipelines[aug_type]
            
            try:
                # 应用增强
                aug_image, aug_boxes = apply_augmentation(image, boxes, aug_pipeline)
                
                if not aug_boxes:
                    continue
                
                # 保存增强后的图像和标签
                aug_filename = f"{os.path.splitext(filename)[0]}_aug_{aug_type}_{i+1}{os.path.splitext(filename)[1]}"
                aug_img_path = os.path.join(augmented_images_dir, aug_filename)
                aug_label_filename = f"{os.path.splitext(filename)[0]}_aug_{aug_type}_{i+1}.txt"
                aug_label_path = os.path.join(augmented_labels_dir, aug_label_filename)
                
                # 保存图像
                cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # 保存标签
                with open(aug_label_path, 'w') as f:
                    for box in aug_boxes:
                        cls_id, x_center, y_center, width, height = box
                        f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
                        aug_stats['by_class'][cls_id] += 1
                
                aug_stats['augmented'] += 1
                aug_stats['by_technique'][aug_type] += 1
                
            except Exception as e:
                logger.error(f"增强图像 {filename} 时出错: {e}")
    
    # 将增强统计信息保存为图表
    visualize_augmentation_stats(aug_stats, class_names, os.path.join('..', 'results', 'visualizations'))
    
    logger.info(f"数据增强完成。原始图像: {aug_stats['original']}, 增强图像: {aug_stats['augmented']}")
    
    # 更新数据配置文件
    data_config_path = os.path.join(data_dir, 'data.yaml')
    augmented_config_path = os.path.join(output_dir, 'data.yaml')
    
    if os.path.exists(data_config_path):
        config = read_data_config(data_config_path)
        config['train'] = os.path.join(output_dir, 'train', 'images')
        config['val'] = os.path.join(output_dir, 'val', 'images')
        config['test'] = os.path.join(output_dir, 'test', 'images')
        
        with open(augmented_config_path, 'w') as f:
            yaml.dump(config, f)
    
    return augmented_config_path, aug_stats

# 可视化数据增强统计信息
def visualize_augmentation_stats(aug_stats, class_names, output_dir):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
    
    # 1. 原始vs增强图像数量
    plt.figure(figsize=(8, 6))
    counts = [aug_stats['original'], aug_stats['augmented']]
    labels = ['原始图像', '增强图像']
    colors = ['#3498db', '#e74c3c']
    
    plt.bar(labels, counts, color=colors)
    for i, count in enumerate(counts):
        plt.text(i, count + 10, str(count), ha='center', fontproperties=font)
    
    plt.title('原始与增强后图像数量对比', fontproperties=font)
    plt.ylabel('图像数量', fontproperties=font)
    plt.savefig(os.path.join(output_dir, 'augmentation_image_counts.png'), dpi=300)
    
    # 2. 各类别增强数量
    plt.figure(figsize=(12, 6))
    classes = [class_names[i] for i in range(len(class_names))]
    counts = [aug_stats['by_class'][i] for i in range(len(class_names))]
    
    plt.bar(classes, counts, color='#2ecc71')
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', fontproperties=font)
    
    plt.title('各类别增强实例数量', fontproperties=font)
    plt.ylabel('实例数量', fontproperties=font)
    plt.xticks(rotation=45, fontproperties=font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'augmentation_by_class.png'), dpi=300)
    
    # 3. 各增强技术使用次数
    plt.figure(figsize=(10, 6))
    techniques = list(aug_stats['by_technique'].keys())
    counts = list(aug_stats['by_technique'].values())
    
    plt.bar(techniques, counts, color='#9b59b6')
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', fontproperties=font)
    
    plt.title('各增强技术使用次数', fontproperties=font)
    plt.ylabel('使用次数', fontproperties=font)
    plt.xticks(rotation=45, fontproperties=font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'augmentation_by_technique.png'), dpi=300)
    
    logger.info(f"增强统计可视化已保存到: {output_dir}")

# 可视化增强效果示例
def visualize_augmentation_examples(data_dir, aug_dir, class_names, output_dir, num_examples=3):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
    
    # 获取原始图像
    train_images_dir = os.path.join(data_dir, 'train', 'images')
    train_labels_dir = os.path.join(data_dir, 'train', 'labels')
    
    # 获取增强后的图像
    aug_images_dir = os.path.join(aug_dir, 'train', 'images')
    aug_labels_dir = os.path.join(aug_dir, 'train', 'labels')
    
    # 查找包含"aug"的文件名
    aug_files = [f for f in os.listdir(aug_images_dir) if 'aug' in f]
    
    # 随机选择几个增强后的图像
    if not aug_files:
        logger.warning("没有找到增强图像")
        return
        
    selected_aug_files = random.sample(aug_files, min(num_examples, len(aug_files)))
    
    # 对于每个增强图像，找到其原始图像
    for i, aug_file in enumerate(selected_aug_files):
        # 获取原始文件名
        orig_file = aug_file.split('_aug_')[0] + os.path.splitext(aug_file)[1]
        
        # 读取原始图像和标签
        orig_img_path = os.path.join(train_images_dir, orig_file)
        orig_label_path = os.path.join(train_labels_dir, os.path.splitext(orig_file)[0] + '.txt')
        
        # 读取增强图像和标签
        aug_img_path = os.path.join(aug_images_dir, aug_file)
        aug_label_path = os.path.join(aug_labels_dir, os.path.splitext(aug_file)[0] + '.txt')
        
        if not (os.path.exists(orig_img_path) and os.path.exists(aug_img_path)):
            continue
            
        # 读取图像
        orig_img = cv2.imread(orig_img_path)
        aug_img = cv2.imread(aug_img_path)
        
        if orig_img is None or aug_img is None:
            continue
            
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        orig_boxes = read_yolo_label(orig_label_path)
        aug_boxes = read_yolo_label(aug_label_path)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 绘制原始图像和边界框
        ax1.imshow(orig_img)
        ax1.set_title('原始图像', fontproperties=font)
        
        for box in orig_boxes:
            cls_id, bbox = yolo_to_pixel(box, orig_img.shape[1], orig_img.shape[0])
            xmin, ymin, xmax, ymax = bbox
            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                    linewidth=2, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
            
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            ax1.text(xmin, ymin - 5, class_name, color='r', fontproperties=font,
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # 绘制增强图像和边界框
        ax2.imshow(aug_img)
        aug_type = aug_file.split('_aug_')[1].split('_')[0]
        ax2.set_title(f'增强图像 ({aug_type})', fontproperties=font)
        
        for box in aug_boxes:
            cls_id, bbox = yolo_to_pixel(box, aug_img.shape[1], aug_img.shape[0])
            xmin, ymin, xmax, ymax = bbox
            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                    linewidth=2, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
            
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            ax2.text(xmin, ymin - 5, class_name, color='r', fontproperties=font,
                    bbox=dict(facecolor='white', alpha=0.7))
        
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'augmentation_example_{i+1}.png'), dpi=300)
        plt.close()
    
    logger.info(f"增强效果示例已保存到: {output_dir}")

def main():
    # 设置路径
    data_dir = os.path.join('..', 'data', 'yolo_format')
    aug_dir = os.path.join('..', 'data', 'yolo_augmented')
    output_dir = os.path.join('..', 'results', 'visualizations')
    
    # 确保输出目录存在
    ensure_dir(aug_dir)
    ensure_dir(os.path.join(aug_dir, 'train', 'images'))
    ensure_dir(os.path.join(aug_dir, 'train', 'labels'))
    ensure_dir(os.path.join(aug_dir, 'val', 'images'))
    ensure_dir(os.path.join(aug_dir, 'val', 'labels'))
    ensure_dir(os.path.join(aug_dir, 'test', 'images'))
    ensure_dir(os.path.join(aug_dir, 'test', 'labels'))
    ensure_dir(output_dir)
    
    # 读取数据配置
    data_config_path = os.path.join(data_dir, 'data.yaml')
    if not os.path.exists(data_config_path):
        logger.error(f"数据配置文件不存在: {data_config_path}")
        return
        
    config = read_data_config(data_config_path)
    class_names = config['names']
    
    logger.info(f"开始数据增强，类别: {class_names}")
    
    # 执行数据增强
    augmented_config_path, aug_stats = augment_dataset(data_dir, aug_dir, class_names, aug_factor=3)
    
    # 可视化增强效果示例
    visualize_augmentation_examples(data_dir, aug_dir, class_names, output_dir)
    
    logger.info(f"数据增强完成。增强后配置文件: {augmented_config_path}")

if __name__ == "__main__":
    main()
````

### 3. 改进的YOLOv10模型

````python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
import logging
import os
from typing import List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('..', 'results', 'logs', 'model.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Conv(nn.Module):
    """标准卷积模块，包含卷积、批归一化和激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """标准瓶颈模块"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C2f(nn.Module):
    """CSP瓶颈模块 (Cross Stage Partial) - 高效版本"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.conv1 = Conv(in_channels, 2 * out_channels, 1, 1)
        self.conv2 = Conv((2 + n) * out_channels, out_channels, 1, 1)
        
        self.blocks = nn.ModuleList([
            Bottleneck(out_channels, out_channels, shortcut, expansion)
            for _ in range(n)
        ])

    def forward(self, x):
        y = list(self.conv1(x).split((self.conv1.conv.out_channels // 2, self.conv1.conv.out_channels // 2), 1))
        for i, block in enumerate(self.blocks):
            y.append(block(y[-1]))
        
        # 使用torch.cat而不是列表相加，避免中间内存分配
        return self.conv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """空间金字塔池化 - 快速版"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2  # 减少通道数以提高效率
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), 1))

class FocalAttention(nn.Module):
    """创新点：Focal注意力机制，增强小目标检测能力"""
    def __init__(self, channels, reduction=8, spatial_scale=2):
        super().__init__()
        self.spatial_scale = spatial_scale  # 控制空间注意力的尺度

        # 通道注意力分支
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力分支
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 动态权重
        self.weight_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 获取动态权重
        weights = self.weight_fc(x)
        
        # 通道注意力
        channel_att = self.channel_gate(x)
        channel_out = x * channel_att
        
        # 空间注意力 - 采用多尺度处理
        h, w = x.size(2), x.size(3)
        # 下采样
        x_down = F.interpolate(x, size=(h//self.spatial_scale, w//self.spatial_scale), 
                              mode='bilinear', align_corners=False)
        spatial_att = self.spatial_gate(x_down)
        # 上采样回原始尺寸
        spatial_att = F.interpolate(spatial_att, size=(h, w), mode='bilinear', align_corners=False)
        spatial_out = x * spatial_att
        
        # 动态融合
        out = channel_out * weights[:, 0:1, :, :] + spatial_out * weights[:, 1:, :, :]
        
        return out

class AdaptiveFeatureFusion(nn.Module):
    """创新点：自适应特征融合，根据特征重要性动态调整融合权重"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, 1, 1)
        
        # 创新点：使用全局和局部特征感知
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.local_gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # 动态融合权重
        self.balance = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        # 特征变换
        x = self.conv(x)
        
        # 全局特征权重
        global_weights = self.global_gate(x)
        
        # 局部特征权重
        local_weights = self.local_gate(x)
        
        # 自适应融合
        weights = self.balance * global_weights + (1 - self.balance) * local_weights
        
        return x * weights

class Detect(nn.Module):
    """YOLOv10检测头 - 创新点：改进的预测网络和标签分配"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # 分类后回归最大数量
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数
        
        # 创新点：改进的分类和回归头
        self.cv = nn.ModuleList([nn.Conv2d(ch[i], self.no, # filepath: c:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\scripts\model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
import logging
import os
from typing import List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('..', 'results', 'logs', 'model.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Conv(nn.Module):
    """标准卷积模块，包含卷积、批归一化和激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """标准瓶颈模块"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C2f(nn.Module):
    """CSP瓶颈模块 (Cross Stage Partial) - 高效版本"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.conv1 = Conv(in_channels, 2 * out_channels, 1, 1)
        self.conv2 = Conv((2 + n) * out_channels, out_channels, 1, 1)
        
        self.blocks = nn.ModuleList([
            Bottleneck(out_channels, out_channels, shortcut, expansion)
            for _ in range(n)
        ])

    def forward(self, x):
        y = list(self.conv1(x).split((self.conv1.conv.out_channels // 2, self.conv1.conv.out_channels // 2), 1))
        for i, block in enumerate(self.blocks):
            y.append(block(y[-1]))
        
        # 使用torch.cat而不是列表相加，避免中间内存分配
        return self.conv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """空间金字塔池化 - 快速版"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2  # 减少通道数以提高效率
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), 1))

class FocalAttention(nn.Module):
    """创新点：Focal注意力机制，增强小目标检测能力"""
    def __init__(self, channels, reduction=8, spatial_scale=2):
        super().__init__()
        self.spatial_scale = spatial_scale  # 控制空间注意力的尺度

        # 通道注意力分支
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力分支
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 动态权重
        self.weight_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 获取动态权重
        weights = self.weight_fc(x)
        
        # 通道注意力
        channel_att = self.channel_gate(x)
        channel_out = x * channel_att
        
        # 空间注意力 - 采用多尺度处理
        h, w = x.size(2), x.size(3)
        # 下采样
        x_down = F.interpolate(x, size=(h//self.spatial_scale, w//self.spatial_scale), 
                              mode='bilinear', align_corners=False)
        spatial_att = self.spatial_gate(x_down)
        # 上采样回原始尺寸
        spatial_att = F.interpolate(spatial_att, size=(h, w), mode='bilinear', align_corners=False)
        spatial_out = x * spatial_att
        
        # 动态融合
        out = channel_out * weights[:, 0:1, :, :] + spatial_out * weights[:, 1:, :, :]
        
        return out

class AdaptiveFeatureFusion(nn.Module):
    """创新点：自适应特征融合，根据特征重要性动态调整融合权重"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, 1, 1)
        
        # 创新点：使用全局和局部特征感知
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.local_gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # 动态融合权重
        self.balance = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        # 特征变换
        x = self.conv(x)
        
        # 全局特征权重
        global_weights = self.global_gate(x)
        
        # 局部特征权重
        local_weights = self.local_gate(x)
        
        # 自适应融合
        weights = self.balance * global_weights + (1 - self.balance) * local_weights
        
        return x * weights

class Detect(nn.Module):
    """YOLOv10检测头 - 创新点：改进的预测网络和标签分配"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # 分类后回归最大数量
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数
        
        # 创新点：改进的分类和回归头
        self.cv = nn.ModuleList([nn.Conv2d(ch[i], self.no, 

找到具有 3 个许可证类型的类似代码

# 完整的YOLO模型实现代码

下面是剩余的完整脚本，包括改进的YOLOv10模型及训练、评估和可视化代码。

## 3. 改进的YOLOv10模型 (model.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ConvBNSiLU(nn.Module):
    """标准卷积块：Conv + BatchNorm + SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GhostConv(nn.Module):
    """Ghost卷积：使用更少的参数生成更多的特征"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = ConvBNSiLU(
            in_channels, init_channels, kernel_size, stride, kernel_size//2
        )
        self.cheap_operation = ConvBNSiLU(
            init_channels, new_channels, 3, 1, 1, groups=init_channels
        )
    
    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        return torch.cat([primary, cheap], dim=1)

class CARAFE(nn.Module):
    """内容感知上采样模块"""
    def __init__(self, channels, scale_factor=2, up_kernel=5):
        super().__init__()
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.kernel_size = up_kernel
        self.kernel_channels = self.kernel_size * self.kernel_size
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, self.kernel_channels * scale_factor * scale_factor, 3, 1, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 生成卷积核权重
        kernel_tensor = self.encoder(x)
        kernel_tensor = kernel_tensor.view(
            b, self.kernel_size * self.kernel_size, 
            self.scale_factor * h, self.scale_factor * w
        )
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.view(
            b, self.kernel_size, self.kernel_size, 
            self.scale_factor * h, self.scale_factor * w
        )
        
        # 上采样输入特征
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        
        # 应用CARAFE操作
        x_unfolded = F.unfold(
            x, kernel_size=self.kernel_size, 
            padding=self.kernel_size//2
        )
        x_unfolded = x_unfolded.view(
            b, c, self.kernel_size * self.kernel_size, 
            self.scale_factor * h, self.scale_factor * w
        )
        
        # 应用生成的卷积核
        out = torch.zeros_like(x)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                idx = i * self.kernel_size + j
                out += x_unfolded[:, :, idx, :, :] * kernel_tensor[:, i, j, :, :].unsqueeze(1)
        
        return out

class DyReLU(nn.Module):
    """动态ReLU激活函数"""
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super().__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        
        # 注意力机制
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)
        # 初始化参数
        self.fc2.bias.data[:k] = 1.0  # 初始化a为1
        self.fc2.bias.data[k:] = 0.0  # 初始化b为0
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        if self.conv_type == '2d':
            batch, channels, height, width = x.size()
            # 全局平均池化
            pooled = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        else:
            batch, channels = x.size()
            pooled = x
            
        # 计算通道注意力
        theta = self.fc1(pooled)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1  # 将值映射到[-1, 1]
        
        # 重塑参数
        a, b = theta[:, :self.k], theta[:, self.k:]
        
        if self.conv_type == '2d':
            a = a.view(batch, self.k, 1, 1)
            b = b.view(batch, self.k, 1, 1)
            
            # 计算动态ReLU
            out = 0
            for i in range(self.k):
                out += torch.clamp(x * a[:, i] + b[:, i], min=0)
            return out
        else:
            a = a.view(batch, self.k, 1)
            b = b.view(batch, self.k, 1)
            
            # 计算动态ReLU
            x = x.unsqueeze(-1)
            out = 0
            for i in range(self.k):
                out += torch.clamp(x * a[:, i] + b[:, i], min=0)
            return out.squeeze(-1)

class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        # 空间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_gate(x)
        channel_att = torch.sigmoid(channel_att)
        x = x * channel_att
        
        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_gate(spatial)
        x = x * spatial_att
        
        return x

class BottleneckCSP(nn.Module):
    """CSP Bottleneck with 增强"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, use_att=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv4 = ConvBNSiLU(2 * hidden_channels, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hidden_channels)
        
        # 添加注意力机制
        self.use_att = use_att
        if use_att:
            self.cbam = CBAM(hidden_channels)
        
        # 主干部分
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0
            ) for _ in range(n)]
        )
    
    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bottlenecks(y1)
        
        if self.use_att:
            y1 = self.cbam(y1)
            
        y2 = self.conv2(x)
        y = torch.cat([y1, y2], dim=1)
        y = self.bn(y)
        y = F.silu(y)
        return self.conv4(y)

class Bottleneck(nn.Module):
    """标准瓶颈结构"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNSiLU(hidden_channels, out_channels, 3, padding=1)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class CrossStagePartialBlock(nn.Module):
    """Enhanced CSP Block with Dynamic Layers"""
    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=True, expansion=0.5, use_att=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # 部分1路径
        self.part1_conv = ConvBNSiLU(in_channels, hidden_channels, 1)
        
        # 部分2路径 (主干)
        self.part2_conv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.part2_bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0) 
            for _ in range(num_bottlenecks)
        ])
        
        if use_att:
            self.attention = CBAM(hidden_channels)
        else:
            self.attention = nn.Identity()
        
        self.part2_conv2 = ConvBNSiLU(hidden_channels, hidden_channels, 1)
        
        # 合并路径
        self.concat_conv = ConvBNSiLU(hidden_channels * 2, out_channels, 1)
        
        # 动态ReLU
        self.act = DyReLU(out_channels)
    
    def forward(self, x):
        # 部分1路径
        part1 = self.part1_conv(x)
        
        # 部分2路径 (主干)
        part2 = self.part2_conv1(x)
        part2 = self.part2_bottlenecks(part2)
        part2 = self.attention(part2)
        part2 = self.part2_conv2(part2)
        
        # 合并并输出
        out = torch.cat([part2, part1], dim=1)
        out = self.concat_conv(out)
        return self.act(out)

class SpatialPyramidPooling(nn.Module):
    """空间金字塔池化"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBNSiLU(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1, 1)
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes
        ])
    
    def forward(self, x):
        x = self.conv1(x)
        features = [x]
        features.extend([maxpool(x) for maxpool in self.maxpools])
        out = torch.cat(features, dim=1)
        return self.conv2(out)

class Focus(nn.Module):
    """Focus模块：将空间维度信息聚焦到通道维度"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super().__init__()
        self.conv = ConvBNSiLU(in_channels * 4, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        # 将图像切分为4个部分并沿通道维拼接
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bottom_left = x[..., 1::2, ::2]
        patch_bottom_right = x[..., 1::2, 1::2]
        x = torch.cat([
            patch_top_left, patch_top_right,
            patch_bottom_left, patch_bottom_right
        ], dim=1)
        return self.conv(x)

class DownsampleBlock(nn.Module):
    """下采样块，降低特征图尺寸"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBNSiLU(in_channels, out_channels, 3, 2, 1)
    
    def forward(self, x):
        return self.conv(x)

class CoorAttention(nn.Module):
    """坐标注意力机制"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(channels // reduction, channels, 1, 1, 0)
        self.conv_w = nn.Conv2d(channels // reduction, channels, 1, 1, 0)
    
    def forward(self, x):
        identity = x
        
        batch, channels, height, width = x.size()
        # 沿高度方向池化
        x_h = self.h_pool(x)
        # 沿宽度方向池化
        x_w = self.w_pool(x).permute(0, 1, 3, 2)
        
        # 合并两个方向的特征
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act1(y)
        
        # 再次分离
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力图
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        # 应用注意力
        out = identity * a_h * a_w
        
        return out

class BiFormerBlock(nn.Module):
    """改进的Transformer Block with Efficient Attention"""
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4):
        super().__init__()
        inner_dim = dim_head * heads
        mlp_hidden_dim = dim * mlp_ratio
        
        # LN for attention
        self.norm1 = nn.LayerNorm(dim)
        
        # Efficient multi-head attention
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attention_scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        
        # Output projection
        self.to_out = nn.Linear(inner_dim, dim)
        
        # LN for MLP
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        # Dynamic weights
        self.dynamic_weights = nn.Parameter(torch.ones(2))
    
    def forward(self, x):
        # 注意：输入 x 形状为 (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        
        # 归一化并计算注意力
        identity = x_flat
        x_ln = self.norm1(x_flat)
        
        # Multi-head attention
        qkv = self.to_qkv(x_ln).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.heads, self.dim_head).transpose(1, 2), qkv)
        
        # 高效稀疏注意力
        attn = (q @ k.transpose(-2, -1)) * self.attention_scale
        attn = attn.softmax(dim=-1)
        
        # 注意力输出
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.to_out(out)
        
        # 第一个残差连接
        weights = F.softmax(self.dynamic_weights, dim=0)
        x_flat = identity * weights[0] + out * weights[1]
        
        # MLP
        identity = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        
        # 第二个残差连接
        x_flat = identity + x_flat
        
        # 恢复原始形状
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x

class FTB(nn.Module):
    """高效特征变换块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = ConvBNSiLU(in_channels, out_channels, 1)
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            out_channels, out_channels, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
        # 注意力机制
        self.se = SEModule(out_channels)
    
    def forward(self, x):
        x = self.conv1x1(x)
        identity = x
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = self.se(x)
        
        return x + identity

class YOLOv10(nn.Module):
    """改进的YOLOv10模型"""
    def __init__(self, num_classes=7, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 主干网络 (Backbone)
        self.backbone = self._create_backbone()
        
        # 特征金字塔网络 (Feature Pyramid Network)
        self.fpn = self._create_fpn()
        
        # 检测头 (Detection Heads)
        self.heads = self._create_heads()
        
        # 锚点尺寸
        self.register_buffer('strides', torch.tensor([8, 16, 32]))
        self.register_buffer('anchors', 
            torch.tensor([
                [[10, 13], [16, 30], [33, 23]],  # 小物体 (P3)
                [[30, 61], [62, 45], [59, 119]],  # 中物体 (P4)
                [[116, 90], [156, 198], [373, 326]]  # 大物体 (P5)
            ]).float()
        )
        
        # 初始化权重
        self._init_weights()
        
    def _create_backbone(self):
        """创建主干网络"""
        return nn.ModuleList([
            # 输入层: 3 -> 32
            Focus(3, 32, 3, 1, 1),
            
            # 下采样1: 32 -> 64
            DownsampleBlock(32, 64),
            CrossStagePartialBlock(64, 64, 1),
            
            # 下采样2: 64 -> 128
            DownsampleBlock(64, 128),
            CrossStagePartialBlock(128, 128, 3),
            
            # 下采样3: 128 -> 256
            DownsampleBlock(128, 256),
            CrossStagePartialBlock(256, 256, 3),
            
            # 下采样4: 256 -> 512
            DownsampleBlock(256, 512),
            CrossStagePartialBlock(512, 512, 1),
            
            # 下采样5: 512 -> 1024
            DownsampleBlock(512, 1024),
            CrossStagePartialBlock(1024, 1024, 1),
            
            # SPP: 1024 -> 1024
            SpatialPyramidPooling(1024, 1024),
        ])
    
    def _create_fpn(self):
        """创建特征金字塔网络"""
        return nn.ModuleList([
            # 上采样路径
            ConvBNSiLU(1024, 512, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            CrossStagePartialBlock(1024, 512, 1),  # 拼接后
            
            ConvBNSiLU(512, 256, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            CrossStagePartialBlock(512, 256, 1),   # 拼接后
            
            # 下采样路径 (特征增强)
            ConvBNSiLU(256, 256, 3, 2, 1),
            CrossStagePartialBlock(512, 512, 1),   # 拼接后
            
            ConvBNSiLU(512, 512, 3, 2, 1),
            CrossStagePartialBlock(1024, 1024, 1)  # 拼接后
        ])
        
    def _create_heads(self):
        """创建检测头"""
        return nn.ModuleList([
            # 小物体检测头 (P3/8)
            nn.Sequential(
                ConvBNSiLU(256, 256, 3, 1, 1),
                nn.Conv2d(256, 3 * (5 + self.num_classes), 1)
            ),
            # 中物体检测头 (P4/16)
            nn.Sequential(
                ConvBNSiLU(512, 512, 3, 1, 1),
                nn.Conv2d(512, 3 * (5 + self.num_classes), 1)
            ),
            # 大物体检测头 (P5/32)
            nn.Sequential(
                ConvBNSiLU(1024, 1024, 3, 1, 1),
                BiFormerBlock(1024),  # 对大物体使用Transformer提取全局上下文
                nn.Conv2d(1024, 3 * (5 + self.num_classes), 1)
            )
        ])
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i in [6, 8, 11]:  # 保存P3, P4, P5特征图
                features.append(x)
        
        # FPN: 自下而上的路径增强
        fpn_features = [features[2]]  # P5
        
        # 上采样路径
        x = self.fpn[0](fpn_features[0])
        x = self.fpn[1](x)
        
        # 拼接P4
        x = torch.cat([x, features[1]], dim=1)
        x = self.fpn[2](x)
        fpn_features.insert(0, x)  # P4
        
        x = self.fpn[3](fpn_features[0])
        x = self.fpn[4](x)
        
        # 拼接P3
        x = torch.cat([x, features[0]], dim=1)
        x = self.fpn[5](x)
        fpn_features.insert(0, x)  # P3
        
        # 下采样路径 (特征增强)
        x = self.fpn[6](fpn_features[0])
        x = torch.cat([x, fpn_features[1]], dim=1)
        x = self.fpn[7](x)
        fpn_features[1] = x  # 增强的P4
        
        x = self.fpn[8](fpn_features[1])
        x = torch.cat([x, fpn_features[2]], dim=1)
        x = self.fpn[9](x)
        fpn_features[2] = x  # 增强的P5
        
        # 检测头
        outputs = []
        for i, head in enumerate(self.heads):
            outputs.append(head(fpn_features[i]))
        
        return outputs
    
    def _make_grid(self, nx=20, ny=20, i=0):
        """生成网格"""
        device = self.anchors.device
        tensor = torch.zeros(1, 3, ny, nx, 5 + self.num_classes, device=device)
        yv, xv = torch.meshgrid([torch.arange(ny, device=device), torch.arange(nx, device=device)])
        grid = torch.stack((xv, yv), 2).expand(1, 3, ny, nx, 2).float()
        anchor_grid = self.anchors[i].clone().view(1, 3, 1, 1, 2).expand(1, 3, ny, nx, 2).to(device)
        
        tensor[..., 0:2] = grid + 0.5  # xy
        tensor[..., 2:4] = anchor_grid  # wh
        return tensor.view(1, 3 * ny * nx, -1)
    
    def process_predictions(self, predictions, img_size):
        """处理预测结果，转换为标准格式"""
        batch_size = predictions[0].shape[0]
        device = predictions[0].device
        
        grids = []
        for i, pred in enumerate(predictions):
            # 获取网格大小
            ny, nx = pred.shape[2:4]
            
            # 重塑预测结果: [B, 3*(5+C), H, W] -> [B, 3, H, W, 5+C]
            pred = pred.view(batch_size, 3, 5 + self.num_classes, ny, nx)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # 应用激活函数
            pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])  # xy
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.anchors[i].to(device)  # wh
            pred[..., 4:] = torch.sigmoid(pred[..., 4:])  # conf, cls
            
            # 计算网格偏移
            stride = img_size / nx
            grid = self._make_grid(nx, ny, i)
            pred[..., 0:2] = (pred[..., 0:2] * 2. - 0.5 + grid[..., 0:2]) * stride
            pred[..., 2:4] = pred[..., 2:4] * stride
            
            # [B, 3*H*W, 5+C]
            grids.append(pred.reshape(batch_size, -1, 5 + self.num_classes))
        
        # 合并所有预测
        return torch.cat(grids, dim=1)
```

## 4. 训练脚本 (train.py)

```python
import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.ops import box_iou

# 导入自定义模块
from model import YOLOv10
from utils import (
    setup_logging, setup_seed, 
    plot_loss_curves, plot_metrics,
    draw_pr_curve, draw_confusion_matrix,
    xyxy2xywh, xywh2xyxy, box_area
)

# 设置路径常量
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
VIS_DIR = os.path.join(RESULTS_DIR, "visualization")

# 创建必要的目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# 配置参数
class Config:
    # 数据集参数
    data_root = os.path.join(DATA_DIR, "insects")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    
    # 模型参数
    img_size = 640
    batch_size = 4
    num_workers = 4
    
    # 训练参数
    epochs = 100
    lr = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    
    # 早停参数
    patience = 5
    
    # 优化器参数
    optimizer = "AdamW"  # "SGD" or "Adam" or "AdamW"
    scheduler = "cosine"  # "step" or "cosine"
    
    # 数据增强
    mosaic_prob = 0.5
    mixup_prob = 0.3
    
    # 损失函数权重
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_cls = 0.5
    lambda_box = 1.0
    
    # 评估参数
    conf_thres = 0.25
    nms_thres = 0.45
    iou_thres = 0.5
    
    # 日志参数
    log_interval = 10
    save_interval = 5
    eval_interval = 1
    
    # 可视化参数
    vis_batch = 2
    vis_samples = 4
    
    # 随机种子
    seed = 42

def get_classes(data_dir):
    """从数据集中获取所有类别"""
    classes = set()
    
    # 遍历所有训练集标注
    ann_dir = os.path.join(data_dir, "train", "annotations")
    for xml_file in os.listdir(ann_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()
        
        for obj in root.findall('object'):
            cls = obj.find('name').text
            classes.add(cls)
    
    return sorted(list(classes))

class InsectsDataset(Dataset):
    """昆虫检测数据集"""
    def __init__(self, root_dir, img_size=640, mode='train', transform=None):
        """
        初始化数据集
        Args:
            root_dir: 数据根目录
            img_size: 图像大小
            mode: 'train', 'val' or 'test'
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        
        # 加载类别
        self.classes = get_classes(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 设置目录
        self.img_dir = os.path.join(root_dir, mode, "images")
        self.ann_dir = os.path.join(root_dir, mode, "annotations")
        
        # 获取所有图像文件名
        self.img_files = [f for f in os.listdir(self.img_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"加载了 {len(self.img_files)} 个 {mode} 图像")
        print(f"类别: {self.classes}")
        
        # 配置数据增强
        self.mosaic_prob = 0.5 if mode == 'train' else 0
        self.mixup_prob = 0.3 if mode == 'train' else 0
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 基础加载
        if np.random.random() < self.mosaic_prob:
            img, targets = self.load_mosaic(idx)
        else:
            img, targets = self.load_image_and_targets(idx)
            
        # Mixup
        if np.random.random() < self.mixup_prob:
            img_mix, targets_mix = self.load_image_and_targets(np.random.randint(0, len(self)))
            r = np.random.beta(8.0, 8.0)  # 混合比例
            img = img * r + img_mix * (1 - r)
            targets = torch.cat([targets, targets_mix], dim=0)
            
        # 应用其他变换
        if self.transform:
            img = self.transform(img)
        
        return img, targets
    
    def load_image_and_targets(self, idx):
        """加载单张图像及其标注"""
        # 加载图像
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取原始尺寸
        h, w = img.shape[:2]
        
        # 缩放系数
        scale_w, scale_h = self.img_size / w, self.img_size / h
        
        # 缩放图像
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 转换为tensor
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float() / 255.0
        
        # 加载标注
        ann_file = os.path.splitext(img_file)[0] + '.xml'
        ann_path = os.path.join(self.ann_dir, ann_file)
        
        boxes = []
        classes = []
        
        if os.path.exists(ann_path):
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                cls = obj.find('name').text
                if cls not in self.class_to_idx:
                    continue
                
                cls_id = self.class_to_idx[cls]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 缩放边界框
                xmin *= scale_w
                ymin *= scale_h
                xmax *= scale_w
                ymax *= scale_h
                
                # 检查边界框有效性
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # 裁剪到图像边界
                xmin = max(0, min(self.img_size - 1, xmin))
                ymin = max(0, min(self.img_size - 1, ymin))
                xmax = max(0, min(self.img_size - 1, xmax))
                ymax = max(0, min(self.img_size - 1, ymax))
                
                # 转换为中心点+宽高格式
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                w = xmax - xmin
                h = ymax - ymin
                
                # 归一化到[0, 1]
                cx /= self.img_size
                cy /= self.img_size
                w /= self.img_size
                h /= self.img_size
                
                boxes.append([cx, cy, w, h])
                classes.append(cls_id)
        
        if len(boxes) == 0:
            # 如果没有目标，返回空标注
            targets = torch.zeros((0, 5))
        else:
            # 组合框和类别，格式: [cls, cx, cy, w, h]
            targets = torch.zeros((len(boxes), 5))
            targets[:, 0] = torch.tensor(classes)
            targets[:, 1:5] = torch.tensor(boxes)
        
        return img, targets
    
    def load_mosaic(self, idx):
        """Mosaic数据增强"""
        # 随机选择其他3张图像
        indices = [idx] + [np.random.randint(0, len(self)) for _ in range(3)]
        
        # 创建mosaic图像
        mosaic_img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        
        # 随机生成拼接点
        cx, cy = self.img_size // 2, self.img_size // 2
        xc = int(np.random.uniform(cx * 0.5, cx * 1.5))
        yc = int(np.random.uniform(cy * 0.5, cy * 1.5))
        
        # 保存所有目标
        all_targets = []
        
        for i, idx in enumerate(indices):
            img, targets = self.load_image_and_targets(idx)
            
            # 获取图像尺寸
            h, w = img.shape[1:]
            
            # 确定放置位置
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size), min(self.img_size, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # 放置图像
            mosaic_img[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]
            
            # 调整边界框坐标
            if len(targets) > 0:
                # 调整中心坐标
                targets_cp = targets.clone()
                targets_cp[:, 1:3] = targets_cp[:, 1:3] * torch.tensor([w, h])
                targets_cp[:, 1] = targets_cp[:, 1] - x1b + x1a
                targets_cp[:, 2] = targets_cp[:, 2] - y1b + y1a
                
                # 调整宽高
                targets_cp[:, 3:5] = targets_cp[:, 3:5] * torch.tensor([w, h])
                
                # 归一化
                targets_cp[:, 1:5] /= self.img_size
                
                # 过滤无效框
                valid_targets = targets_cp.clone()
                valid_targets[:, 1:3] = torch.clamp(valid_targets[:, 1:3], 0, 1)
                
                # 添加到目标列表
                all_targets.append(valid_targets)
        
        # 合并所有目标
        if len(all_targets) > 0:
            all_targets = torch.cat(all_targets, dim=0)
            # 过滤掉超出边界或太小的框
            valid = (all_targets[:, 3] > 0.01) & (all_targets[:, 4] > 0.01)
            all_targets = all_targets[valid]
        else:
            all_targets = torch.zeros((0, 5))
        
        return mosaic_img, all_targets

# 损失函数定义
class YOLOLoss(nn.Module):
    """改进的YOLO损失函数"""
    def __init__(self, num_classes, anchors, img_size=640, 
                 lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=0.5, lambda_box=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 损失权重
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        
        # 转换格式: [s, m, l] x 3 anchors
        self.register_buffer('anchors', anchors)
        self.strides = torch.tensor([8, 16, 32])
        
        # 定义损失函数
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
        # 对焦损失 (Focal Loss) 参数
        self.alpha = 0.25
        self.gamma = 2.0
        
    def forward(self, predictions, targets):
        """计算损失
        Args:
            predictions: 模型输出，格式为3个特征图的列表
            targets: 目标，格式为[batch_idx, class, x, y, w, h]
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        
        # 初始化损失
        loss_obj = torch.zeros(1, device=device)
        loss_noobj = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)
        
        # 按批次处理targets
        targets_by_batch = [[] for _ in range(batch_size)]
        for target_idx in range(len(targets)):
            batch_idx = 0  # 单个批次的情况
            if len(targets) > 1:  # 处理多个批次
                if len(targets.shape) > 1 and targets.shape[1] > 0:
                    batch_idx = int(targets[target_idx, 0])
            if batch_idx < batch_size:  # 确保batch_idx有效
                targets_by_batch[batch_idx].append(targets[target_idx])
        
        # 处理每个特征图
        for level, pred in enumerate(predictions):
            # 特征图大小
            bs, _, h, w = pred.shape
            
            # 重塑预测结果: [B, 3*(5+C), H, W] -> [B, 3, H, W, 5+C]
            pred = pred.view(bs, 3, 5 + self.num_classes, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # 获取当前特征图的anchors和stride
            anchor_vec = self.anchors[level]
            stride = self.strides[level]
            
            # 创建网格
            grid_y, grid_x = torch.meshgrid([torch.arange(h, device=device), 
                                            torch.arange(w, device=device)])
            grid = torch.stack((grid_x, grid_y), 2).expand(1, 3, h, w, 2).float()
            
            # 输出预测的组件
            pred_xy = torch.sigmoid(pred[..., 0:2])  # 中心点相对网格的偏移 (0-1)
            pred_wh = torch.exp(pred[..., 2:4]) * anchor_vec.unsqueeze(2).unsqueeze(3)  # 宽高相对anchor的缩放
            pred_obj = torch.sigmoid(pred[..., 4:5])  # 目标置信度
            pred_cls = torch.sigmoid(pred[..., 5:])  # 类别概率
            
            # 计算预测框绝对坐标 (0-640)
            pred_boxes = torch.zeros_like(pred[..., :4])
            pred_boxes[..., 0:2] = (pred_xy + grid) * stride  # 绝对xy
            pred_boxes[..., 2:4] = pred_wh * stride  # 绝对wh
            
            # 处理每个batch
            for batch_idx in range(batch_size):
                batch_targets = targets_by_batch[batch_idx]
                if not batch_targets:  # 如果该batch没有目标
                    continue
                
                targets_tensor = torch.stack(batch_targets)
                
                # 提取目标的坐标和类别
                tcls = targets_tensor[:, 0].long()
                tx = targets_tensor[:, 1] * self.img_size
                ty = targets_tensor[:, 2] * self.img_size
                tw = targets_tensor[:, 3] * self.img_size
                th = targets_tensor[:, 4] * self.img_size
                
                # 过滤掉宽高太小的目标
                valid = (tw > 1) & (th > 1)
                if not valid.any():
                    continue
                
                tcls = tcls[valid]
                tx = tx[valid]
                ty = ty[valid]
                tw = tw[valid]
                th = th[valid]
                
                # 将目标转换为当前特征图上的坐标
                gx = tx / stride
                gy = ty / stride
                gw = tw / stride
                gh = th / stride
                
                # 计算网格索引
                gi = gx.long().clamp(0, w - 1)
                gj = gy.long().clamp(0, h - 1)
                
                # 计算最佳匹配的anchor索引
                target_boxes = torch.stack([torch.zeros_like(gx), torch.zeros_like(gy), gw, gh], 1)
                anchor_shapes = torch.cat((torch.zeros(len(anchor_vec), 2), anchor_vec), 1)
                
                # 计算IOU以找到最佳anchor
                anch_ious = box_iou(
                    xyxy2xywh(target_boxes), 
                    xyxy2xywh(anchor_shapes)
                )
                best_ious, best_n = anch_ious.max(1)
                
                # 构建目标张量
                for t_idx in range(len(tcls)):
                    c = tcls[t_idx]
                    i, j = gi[t_idx], gj[t_idx]
                    a = best_n[t_idx]
                    
                    # 目标坐标
                    gx_t, gy_t = gx[t_idx] - gi[t_idx], gy[t_idx] - gj[t_idx]
                    gw_t, gh_t = gw[t_idx], gh[t_idx]
                    
                    # 目标框中心偏移和宽高
                    target_xy = torch.tensor([gx_t, gy_t], device=device)
                    target_wh = torch.tensor([gw_t, gh_t], device=device) / anchor_vec[a]
                    
                    # 更新各项损失
                    # 置信度损失 (对象和非对象)
                    obj_mask = torch.zeros_like(pred_obj[batch_idx])
                    obj_mask[a, j, i] = 1
                    noobj_mask = 1 - obj_mask
                    
                    # 对象损失 (置信度)
                    loss_obj += self.bce(
                        pred_obj[batch_idx], 
                        obj_mask
                    ).sum() * self.lambda_obj
                    
                    # 非对象损失 (置信度)
                    loss_noobj += self.bce(
                        pred_obj[batch_idx], 
                        torch.zeros_like(pred_obj[batch_idx])
                    ).sum() * self.lambda_noobj * noobj_mask
                    
                    # 框坐标损失
                    pred_xy_t = pred_xy[batch_idx, a, j, i]
                    loss_xy = self.mse(pred_xy_t, target_xy).sum() * self.lambda_box
                    
                    # 转换为对数空间计算宽高损失
                    pred_wh_t = torch.log(pred_wh[batch_idx, a, j, i] / anchor_vec[a] + 1e-6)
                    target_wh = torch.log(target_wh + 1e-6)
                    loss_wh = self.mse(pred_wh_t, target_wh).sum() * self.lambda_box
                    
                    loss_box += loss_xy + loss_wh
                    
                    # 类别损失
                    if self.num_classes > 1:  # 多类别情况
                        target_cls = torch.zeros_like(pred_cls[batch_idx, a, j, i])
                        target_cls[c] = 1
                        loss_cls += self.focal_loss(
                            pred_cls[batch_idx, a, j, i], 
                            target_cls
                        ).sum() * self.lambda_cls
        
        # 归一化损失
        num_targets = sum(len(t) for t in targets_by_batch) + 1e-6  # 防止除零
        loss_box /= num_targets
        loss_obj /= num_targets
        loss_noobj /= batch_size * 3 * h * w  # 归一化非对象损失
        loss_cls /= num_targets
        
        # 总损失
        total_loss = loss_box + loss_obj + loss_noobj + loss_cls
        
        # 返回各项损失
        return total_loss, {
            'box': loss_box.item(),
            'obj': loss_obj.item(),
            'noobj': loss_noobj.item(),
            'cls': loss_cls.item(),
            'total': total_loss.item()
        }
    
    def focal_loss(self, pred, target):
        """焦点损失"""
        # 二值交叉熵
        bce = self.bce(pred, target)
        
        # 计算pt
        pt = torch.exp(-bce)
        # 计算焦点损失
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        
        return focal_loss

# 训练函数
def train(cfg):
    """训练主函数"""
    setup_seed(cfg.seed)
    logger = setup_logging(os.path.join(LOGS_DIR, "train.log"))
    
    # 获取类别
    classes = get_classes(cfg.data_root)
    num_classes = len(classes)
    logger.info(f"类别数: {num_classes}, 类别: {classes}")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
    ])
    
    # 创建数据集
    train_dataset = InsectsDataset(
        cfg.data_root, cfg.img_size, 'train', train_transform
    )
    val_dataset = InsectsDataset(
        cfg.data_root, cfg.img_size, 'val', None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 创建模型
    model = YOLOv10(num_classes=num_classes, img_size=cfg.img_size)
    model.cuda()
    
    # 记录模型信息
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {pytorch_total_params / 1e6:.2f}M")
    
    # 创建损失函数
    criterion = YOLOLoss(
        num_classes=num_classes,
        anchors=model.anchors,
        img_size=cfg.img_size,
        lambda_obj=cfg.lambda_obj,
        lambda# filepath: c:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\scripts\train.py
import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.ops import box_iou

# 导入自定义模块
from model import YOLOv10
from utils import (
    setup_logging, setup_seed, 
    plot_loss_curves, plot_metrics,
    draw_pr_curve, draw_confusion_matrix,
    xyxy2xywh, xywh2xyxy, box_area
)

# 设置路径常量
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
VIS_DIR = os.path.join(RESULTS_DIR, "visualization")

# 创建必要的目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# 配置参数
class Config:
    # 数据集参数
    data_root = os.path.join(DATA_DIR, "insects")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    
    # 模型参数
    img_size = 640
    batch_size = 4
    num_workers = 4
    
    # 训练参数
    epochs = 100
    lr = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    
    # 早停参数
    patience = 5
    
    # 优化器参数
    optimizer = "AdamW"  # "SGD" or "Adam" or "AdamW"
    scheduler = "cosine"  # "step" or "cosine"
    
    # 数据增强
    mosaic_prob = 0.5
    mixup_prob = 0.3
    
    # 损失函数权重
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_cls = 0.5
    lambda_box = 1.0
    
    # 评估参数
    conf_thres = 0.25
    nms_thres = 0.45
    iou_thres = 0.5
    
    # 日志参数
    log_interval = 10
    save_interval = 5
    eval_interval = 1
    
    # 可视化参数
    vis_batch = 2
    vis_samples = 4
    
    # 随机种子
    seed = 42

def get_classes(data_dir):
    """从数据集中获取所有类别"""
    classes = set()
    
    # 遍历所有训练集标注
    ann_dir = os.path.join(data_dir, "train", "annotations")
    for xml_file in os.listdir(ann_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()
        
        for obj in root.findall('object'):
            cls = obj.find('name').text
            classes.add(cls)
    
    return sorted(list(classes))

class InsectsDataset(Dataset):
    """昆虫检测数据集"""
    def __init__(self, root_dir, img_size=640, mode='train', transform=None):
        """
        初始化数据集
        Args:
            root_dir: 数据根目录
            img_size: 图像大小
            mode: 'train', 'val' or 'test'
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        
        # 加载类别
        self.classes = get_classes(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 设置目录
        self.img_dir = os.path.join(root_dir, mode, "images")
        self.ann_dir = os.path.join(root_dir, mode, "annotations")
        
        # 获取所有图像文件名
        self.img_files = [f for f in os.listdir(self.img_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"加载了 {len(self.img_files)} 个 {mode} 图像")
        print(f"类别: {self.classes}")
        
        # 配置数据增强
        self.mosaic_prob = 0.5 if mode == 'train' else 0
        self.mixup_prob = 0.3 if mode == 'train' else 0
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 基础加载
        if np.random.random() < self.mosaic_prob:
            img, targets = self.load_mosaic(idx)
        else:
            img, targets = self.load_image_and_targets(idx)
            
        # Mixup
        if np.random.random() < self.mixup_prob:
            img_mix, targets_mix = self.load_image_and_targets(np.random.randint(0, len(self)))
            r = np.random.beta(8.0, 8.0)  # 混合比例
            img = img * r + img_mix * (1 - r)
            targets = torch.cat([targets, targets_mix], dim=0)
            
        # 应用其他变换
        if self.transform:
            img = self.transform(img)
        
        return img, targets
    
    def load_image_and_targets(self, idx):
        """加载单张图像及其标注"""
        # 加载图像
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取原始尺寸
        h, w = img.shape[:2]
        
        # 缩放系数
        scale_w, scale_h = self.img_size / w, self.img_size / h
        
        # 缩放图像
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 转换为tensor
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float() / 255.0
        
        # 加载标注
        ann_file = os.path.splitext(img_file)[0] + '.xml'
        ann_path = os.path.join(self.ann_dir, ann_file)
        
        boxes = []
        classes = []
        
        if os.path.exists(ann_path):
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                cls = obj.find('name').text
                if cls not in self.class_to_idx:
                    continue
                
                cls_id = self.class_to_idx[cls]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 缩放边界框
                xmin *= scale_w
                ymin *= scale_h
                xmax *= scale_w
                ymax *= scale_h
                
                # 检查边界框有效性
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # 裁剪到图像边界
                xmin = max(0, min(self.img_size - 1, xmin))
                ymin = max(0, min(self.img_size - 1, ymin))
                xmax = max(0, min(self.img_size - 1, xmax))
                ymax = max(0, min(self.img_size - 1, ymax))
                
                # 转换为中心点+宽高格式
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                w = xmax - xmin
                h = ymax - ymin
                
                # 归一化到[0, 1]
                cx /= self.img_size
                cy /= self.img_size
                w /= self.img_size
                h /= self.img_size
                
                boxes.append([cx, cy, w, h])
                classes.append(cls_id)
        
        if len(boxes) == 0:
            # 如果没有目标，返回空标注
            targets = torch.zeros((0, 5))
        else:
            # 组合框和类别，格式: [cls, cx, cy, w, h]
            targets = torch.zeros((len(boxes), 5))
            targets[:, 0] = torch.tensor(classes)
            targets[:, 1:5] = torch.tensor(boxes)
        
        return img, targets
    
    def load_mosaic(self, idx):
        """Mosaic数据增强"""
        # 随机选择其他3张图像
        indices = [idx] + [np.random.randint(0, len(self)) for _ in range(3)]
        
        # 创建mosaic图像
        mosaic_img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        
        # 随机生成拼接点
        cx, cy = self.img_size // 2, self.img_size // 2
        xc = int(np.random.uniform(cx * 0.5, cx * 1.5))
        yc = int(np.random.uniform(cy * 0.5, cy * 1.5))
        
        # 保存所有目标
        all_targets = []
        
        for i, idx in enumerate(indices):
            img, targets = self.load_image_and_targets(idx)
            
            # 获取图像尺寸
            h, w = img.shape[1:]
            
            # 确定放置位置
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size), min(self.img_size, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # 放置图像
            mosaic_img[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]
            
            # 调整边界框坐标
            if len(targets) > 0:
                # 调整中心坐标
                targets_cp = targets.clone()
                targets_cp[:, 1:3] = targets_cp[:, 1:3] * torch.tensor([w, h])
                targets_cp[:, 1] = targets_cp[:, 1] - x1b + x1a
                targets_cp[:, 2] = targets_cp[:, 2] - y1b + y1a
                
                # 调整宽高
                targets_cp[:, 3:5] = targets_cp[:, 3:5] * torch.tensor([w, h])
                
                # 归一化
                targets_cp[:, 1:5] /= self.img_size
                
                # 过滤无效框
                valid_targets = targets_cp.clone()
                valid_targets[:, 1:3] = torch.clamp(valid_targets[:, 1:3], 0, 1)
                
                # 添加到目标列表
                all_targets.append(valid_targets)
        
        # 合并所有目标
        if len(all_targets) > 0:
            all_targets = torch.cat(all_targets, dim=0)
            # 过滤掉超出边界或太小的框
            valid = (all_targets[:, 3] > 0.01) & (all_targets[:, 4] > 0.01)
            all_targets = all_targets[valid]
        else:
            all_targets = torch.zeros((0, 5))
        
        return mosaic_img, all_targets

# 损失函数定义
class YOLOLoss(nn.Module):
    """改进的YOLO损失函数"""
    def __init__(self, num_classes, anchors, img_size=640, 
                 lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=0.5, lambda_box=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 损失权重
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        
        # 转换格式: [s, m, l] x 3 anchors
        self.register_buffer('anchors', anchors)
        self.strides = torch.tensor([8, 16, 32])
        
        # 定义损失函数
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
        # 对焦损失 (Focal Loss) 参数
        self.alpha = 0.25
        self.gamma = 2.0
        
    def forward(self, predictions, targets):
        """计算损失
        Args:
            predictions: 模型输出，格式为3个特征图的列表
            targets: 目标，格式为[batch_idx, class, x, y, w, h]
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        
        # 初始化损失
        loss_obj = torch.zeros(1, device=device)
        loss_noobj = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)
        
        # 按批次处理targets
        targets_by_batch = [[] for _ in range(batch_size)]
        for target_idx in range(len(targets)):
            batch_idx = 0  # 单个批次的情况
            if len(targets) > 1:  # 处理多个批次
                if len(targets.shape) > 1 and targets.shape[1] > 0:
                    batch_idx = int(targets[target_idx, 0])
            if batch_idx < batch_size:  # 确保batch_idx有效
                targets_by_batch[batch_idx].append(targets[target_idx])
        
        # 处理每个特征图
        for level, pred in enumerate(predictions):
            # 特征图大小
            bs, _, h, w = pred.shape
            
            # 重塑预测结果: [B, 3*(5+C), H, W] -> [B, 3, H, W, 5+C]
            pred = pred.view(bs, 3, 5 + self.num_classes, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # 获取当前特征图的anchors和stride
            anchor_vec = self.anchors[level]
            stride = self.strides[level]
            
            # 创建网格
            grid_y, grid_x = torch.meshgrid([torch.arange(h, device=device), 
                                            torch.arange(w, device=device)])
            grid = torch.stack((grid_x, grid_y), 2).expand(1, 3, h, w, 2).float()
            
            # 输出预测的组件
            pred_xy = torch.sigmoid(pred[..., 0:2])  # 中心点相对网格的偏移 (0-1)
            pred_wh = torch.exp(pred[..., 2:4]) * anchor_vec.unsqueeze(2).unsqueeze(3)  # 宽高相对anchor的缩放
            pred_obj = torch.sigmoid(pred[..., 4:5])  # 目标置信度
            pred_cls = torch.sigmoid(pred[..., 5:])  # 类别概率
            
            # 计算预测框绝对坐标 (0-640)
            pred_boxes = torch.zeros_like(pred[..., :4])
            pred_boxes[..., 0:2] = (pred_xy + grid) * stride  # 绝对xy
            pred_boxes[..., 2:4] = pred_wh * stride  # 绝对wh
            
            # 处理每个batch
            for batch_idx in range(batch_size):
                batch_targets = targets_by_batch[batch_idx]
                if not batch_targets:  # 如果该batch没有目标
                    continue
                
                targets_tensor = torch.stack(batch_targets)
                
                # 提取目标的坐标和类别
                tcls = targets_tensor[:, 0].long()
                tx = targets_tensor[:, 1] * self.img_size
                ty = targets_tensor[:, 2] * self.img_size
                tw = targets_tensor[:, 3] * self.img_size
                th = targets_tensor[:, 4] * self.img_size
                
                # 过滤掉宽高太小的目标
                valid = (tw > 1) & (th > 1)
                if not valid.any():
                    continue
                
                tcls = tcls[valid]
                tx = tx[valid]
                ty = ty[valid]
                tw = tw[valid]
                th = th[valid]
                
                # 将目标转换为当前特征图上的坐标
                gx = tx / stride
                gy = ty / stride
                gw = tw / stride
                gh = th / stride
                
                # 计算网格索引
                gi = gx.long().clamp(0, w - 1)
                gj = gy.long().clamp(0, h - 1)
                
                # 计算最佳匹配的anchor索引
                target_boxes = torch.stack([torch.zeros_like(gx), torch.zeros_like(gy), gw, gh], 1)
                anchor_shapes = torch.cat((torch.zeros(len(anchor_vec), 2), anchor_vec), 1)
                
                # 计算IOU以找到最佳anchor
                anch_ious = box_iou(
                    xyxy2xywh(target_boxes), 
                    xyxy2xywh(anchor_shapes)
                )
                best_ious, best_n = anch_ious.max(1)
                
                # 构建目标张量
                for t_idx in range(len(tcls)):
                    c = tcls[t_idx]
                    i, j = gi[t_idx], gj[t_idx]
                    a = best_n[t_idx]
                    
                    # 目标坐标
                    gx_t, gy_t = gx[t_idx] - gi[t_idx], gy[t_idx] - gj[t_idx]
                    gw_t, gh_t = gw[t_idx], gh[t_idx]
                    
                    # 目标框中心偏移和宽高
                    target_xy = torch.tensor([gx_t, gy_t], device=device)
                    target_wh = torch.tensor([gw_t, gh_t], device=device) / anchor_vec[a]
                    
                    # 更新各项损失
                    # 置信度损失 (对象和非对象)
                    obj_mask = torch.zeros_like(pred_obj[batch_idx])
                    obj_mask[a, j, i] = 1
                    noobj_mask = 1 - obj_mask
                    
                    # 对象损失 (置信度)
                    loss_obj += self.bce(
                        pred_obj[batch_idx], 
                        obj_mask
                    ).sum() * self.lambda_obj
                    
                    # 非对象损失 (置信度)
                    loss_noobj += self.bce(
                        pred_obj[batch_idx], 
                        torch.zeros_like(pred_obj[batch_idx])
                    ).sum() * self.lambda_noobj * noobj_mask
                    
                    # 框坐标损失
                    pred_xy_t = pred_xy[batch_idx, a, j, i]
                    loss_xy = self.mse(pred_xy_t, target_xy).sum() * self.lambda_box
                    
                    # 转换为对数空间计算宽高损失
                    pred_wh_t = torch.log(pred_wh[batch_idx, a, j, i] / anchor_vec[a] + 1e-6)
                    target_wh = torch.log(target_wh + 1e-6)
                    loss_wh = self.mse(pred_wh_t, target_wh).sum() * self.lambda_box
                    
                    loss_box += loss_xy + loss_wh
                    
                    # 类别损失
                    if self.num_classes > 1:  # 多类别情况
                        target_cls = torch.zeros_like(pred_cls[batch_idx, a, j, i])
                        target_cls[c] = 1
                        loss_cls += self.focal_loss(
                            pred_cls[batch_idx, a, j, i], 
                            target_cls
                        ).sum() * self.lambda_cls
        
        # 归一化损失
        num_targets = sum(len(t) for t in targets_by_batch) + 1e-6  # 防止除零
        loss_box /= num_targets
        loss_obj /= num_targets
        loss_noobj /= batch_size * 3 * h * w  # 归一化非对象损失
        loss_cls /= num_targets
        
        # 总损失
        total_loss = loss_box + loss_obj + loss_noobj + loss_cls
        
        # 返回各项损失
        return total_loss, {
            'box': loss_box.item(),
            'obj': loss_obj.item(),
            'noobj': loss_noobj.item(),
            'cls': loss_cls.item(),
            'total': total_loss.item()
        }
    
    def focal_loss(self, pred, target):
        """焦点损失"""
        # 二值交叉熵
        bce = self.bce(pred, target)
        
        # 计算pt
        pt = torch.exp(-bce)
        # 计算焦点损失
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        
        return focal_loss

# 训练函数
def train(cfg):
    """训练主函数"""
    setup_seed(cfg.seed)
    logger = setup_logging(os.path.join(LOGS_DIR, "train.log"))
    
    # 获取类别
    classes = get_classes(cfg.data_root)
    num_classes = len(classes)
    logger.info(f"类别数: {num_classes}, 类别: {classes}")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
    ])
    
    # 创建数据集
    train_dataset = InsectsDataset(
        cfg.data_root, cfg.img_size, 'train', train_transform
    )
    val_dataset = InsectsDataset(
        cfg.data_root, cfg.img_size, 'val', None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 创建模型
    model = YOLOv10(num_classes=num_classes, img_size=cfg.img_size)
    model.cuda()
    
    # 记录模型信息
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {pytorch_total_params / 1e6:.2f}M")
    
    # 创建损失函数
    criterion = YOLOLoss(
        num_classes=num_classes,
        anchors=model.anchors,
        img_size=cfg.img_size,
        lambda_obj=cfg.lambda_obj,
        lambda

找到具有 3 个许可证类型的类似代码



ot support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  19%|█▉        | 6/31 [00:00<00:02,  8.77it/s]WARNING Model does not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  26%|██▌       | 8/31 [00:00<00:02, 10.24it/s]WARNING Model does not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  32%|███▏      | 10/31 [00:01<00:01, 11.50it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  39%|███▊      | 12/31 [00:01<00:01, 12.07it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  45%|████▌     | 14/31 [00:01<00:01, 12.47it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  52%|█████▏    | 16/31 [00:01<00:01, 12.79it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  58%|█████▊    | 18/31 [00:01<00:01, 12.92it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  65%|██████▍   | 20/31 [00:01<00:00, 12.78it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  71%|███████   | 22/31 [00:01<00:00, 13.21it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  77%|███████▋  | 24/31 [00:02<00:00, 13.41it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  84%|████████▍ | 26/31 [00:02<00:00, 13.47it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  90%|█████████ | 28/31 [00:02<00:00, 13.18it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
WARNING Model does not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  97%|█████████▋| 30/31 [00:02<00:00, 13.10it/s]WARNING Model does 
not support 'augment=True', reverting to single-scale prediction.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 31/31 [00:02<00:00, 11.72it/s]
                   all        245       1856      0.808      0.789      0.851      0.656



03：
2025-05-24 11:44:36,714 - INFO - 开始评估模型
Ultralytics 8.3.126  Python-3.11.0 torch-2.7.0+cu128 CUDA:0 (NVIDIA GeForce RTX 2050, 4096MiB)
YOLOv10n summary (fused): 102 layers, 2,266,533 parameters, 0 gradients, 6.5 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 427.7106.4 MB/s, size: 476.4 KB)
val: Scanning C:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\data\labels\val.cache... 245 images, 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 16/16 [00:06<00:00,  2.53it/s]
                   all        245       1856      0.811      0.786      0.829      0.652
               Boerner        201        318      0.891      0.931      0.949      0.819
               Leconte        194        594       0.93      0.909      0.946      0.737
              Linnaeus        243        292      0.807       0.63      0.752      0.627
            acuminatus        235        235      0.931      0.672      0.821      0.642
               armandi        219        231       0.71      0.668      0.715      0.566
            coleoptera        181        186      0.595      0.909      0.792      0.524
Speed: 1.5ms preprocess, 10.3ms inference, 0.0ms loss, 0.3ms postprocess per image
Saving C:\Users\lijie\Desktop\\CV-work\ \2-\cv-target-detect\yolo_code\results\metrics\evaluation4\predictions.json...
Results saved to C:\Users\lijie\Desktop\\CV-work\ \2-\cv-target-detect\yolo_code\results\metrics\evaluation4
2025-05-24 11:45:17,880 - INFO - 评估完成
2025-05-24 11:45:17,881 - ERROR - 保存指标失败: only length-1 arrays can be converted to Python scalars
2025-05-24 11:45:17,881 - ERROR - Traceback (most recent call last):
  File "c:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\scripts\03_evaluate_model.py", line 461, in 
main
    'precision': float(metrics.p),
                 ^^^^^^^^^^^^^^^^
TypeError: only length-1 arrays can be converted to Python scalars

2025-05-24 11:45:17,882 - INFO - 复制评估图表: confusion_matrix.png
2025-05-24 11:45:17,885 - INFO - 复制评估图表: confusion_matrix_normalized.png
2025-05-24 11:45:17,887 - INFO - 复制评估图表: F1_curve.png
2025-05-24 11:45:17,889 - INFO - 复制评估图表: PR_curve.png
2025-05-24 11:45:17,891 - INFO - 复制评估图表: P_curve.png
2025-05-24 11:45:17,893 - INFO - 复制评估图表: R_curve.png
2025-05-24 11:45:17,894 - INFO - 可视化预测结果

image 1/1 C:\Users\lijie\Desktop\\CV-work\ \2-\cv-target-detect\yolo_code\data\images\val\2490.jpeg: 640x640 5 Lecontes, 1 Linnaeus, 2 acuminatuss, 1 armandi, 2 coleopteras, 33.7ms
Speed: 4.7ms preprocess, 33.7ms inference, 1.5ms postprocess per image at shape (1, 3, 640, 640)
已保存预测可视化: C:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\results\predictions\pred_vs_true_2490.png

image 1/1 C:\Users\lijie\Desktop\\CV-work\ \2-\cv-target-detect\yolo_code\data\images\val\1992.jpeg: 640x640 1 Boerner, 2 Lecontes, 2 Linnaeuss, 1 coleoptera, 1 linnaeus, 33.9ms
Speed: 4.3ms preprocess, 33.9ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 640)
已保存预测可视化: C:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\results\predictions\pred_vs_true_1992.png

image 1/1 C:\Users\lijie\Desktop\\CV-work\ \2-\cv-target-detect\yolo_code\data\images\val\3134.jpeg: 640x640 3 Boerners, 2 Linnaeuss, 1 acuminatus, 2 armandis, 2 coleopteras, 72.2ms
Speed: 6.4ms preprocess, 72.2ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 640)
已保存预测可视化: C:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\results\predictions\pred_vs_true_3134.png

image 1/1 C:\Users\lijie\Desktop\\CV-work\ \2-\cv-target-detect\yolo_code\data\images\val\3159.jpeg: 640x640 3 Boerners, 1 Linnaeus, 1 acuminatus, 1 armandi, 2 coleopteras, 144.5ms
Speed: 9.4ms preprocess, 144.5ms inference, 1.2ms postprocess per image at shape (1, 3, 640, 640)
已保存预测可视化: C:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\results\predictions\pred_vs_true_3159.png

image 1/1 C:\Users\lijie\Desktop\\CV-work\ \2-\cv-target-detect\yolo_code\data\images\val\2237.jpeg: 640x640 1 Boerner, 3 Lecontes, 1 Linnaeus, 1 acuminatus, 1 armandi, 1 coleoptera, 12.3ms
Speed: 3.5ms preprocess, 12.3ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 640)
已保存预测可视化: C:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\results\predictions\pred_vs_true_2237.png
2025-05-24 11:45:24,041 - INFO - 预测可视化已保存至: C:\Users\lijie\Desktop\大三下资料\CV-work\上机课 四个实验报告\实验2-目标检测\cv-target-detect\yolo_code\results\predictions
2025-05-24 11:45:24,041 - INFO - 评估完成，所有结果已保存