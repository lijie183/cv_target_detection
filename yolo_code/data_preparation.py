import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import warnings

# 忽略版本检查警告
warnings.filterwarnings("ignore", category=UserWarning, message="Error fetching version info")

from config import *

class VOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None, is_test=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_test = is_test
        
        # 根据image_set确定图像ID列表
        self.img_ids = []
        list_file = self.root_dir / "ImageSets" / "Main" / f"{image_set}.txt"
        
        with open(list_file, 'r') as f:
            for line in f:
                # 处理行，提取图像ID
                img_id = line.strip().split()[0]
                self.img_ids.append(img_id)
        
        # 类别名称到索引的映射
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        
        # 读取图像
        img_path = self.root_dir / "JPEGImages" / f"{img_id}.jpg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        
        if not self.is_test:
            # 读取标注
            anno_path = self.root_dir / "Annotations" / f"{img_id}.xml"
            boxes, labels = self._parse_annotation(anno_path)
            
            # 应用变换
            if self.transform:
                transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            
            # 转换为torch张量
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # 创建目标字典
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([index]),
                'orig_size': torch.as_tensor([height, width], dtype=torch.int64)
            }
            
            return img, target, img_id
        else:
            # 测试模式，只返回图像
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            
            return img, img_id, (height, width)
    
    def _parse_annotation(self, anno_path):
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:  # 跳过difficult=1的对象
                continue
                
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 归一化坐标
            img_width = float(root.find('size/width').text)
            img_height = float(root.find('size/height').text)
            
            # YOLO格式：x_center, y_center, width, height (归一化)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            boxes.append([x_center, y_center, width, height])
            labels.append(self.class_to_idx[name])
        
        return boxes, labels

def create_data_loaders():
    # 训练数据增强
    train_transform = A.Compose([
        # 修改为使用元组形式的参数
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.1, 1.0)),
        # 确保 ColorJitter 也使用正确的参数形式
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # 验证数据增强
    val_transform = A.Compose([
        # 修改为使用元组形式的参数
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # 测试数据增强
    test_transform = A.Compose([
        # 修改为使用元组形式的参数
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # 创建数据集
    train_dataset = VOCDataset(
        root_dir=TRAIN_DATA_PATH,
        image_set='train',
        transform=train_transform
    )
    
    val_dataset = VOCDataset(
        root_dir=TRAIN_DATA_PATH,
        image_set='val',
        transform=val_transform
    )
    
    test_dataset = VOCDataset(
        root_dir=TEST_DATA_PATH,
        image_set='test',
        transform=test_transform,
        is_test=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_test_fn
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """自定义批处理函数，处理不同大小的图像和标注"""
    images = []
    targets = []
    img_ids = []
    
    for img, target, img_id in batch:
        images.append(img)
        targets.append(target)
        img_ids.append(img_id)
    
    return images, targets, img_ids

def collate_test_fn(batch):
    """测试集的批处理函数"""
    images = []
    img_ids = []
    orig_sizes = []
    
    for img, img_id, orig_size in batch:
        images.append(img)
        img_ids.append(img_id)
        orig_sizes.append(orig_size)
    
    images = torch.stack(images, 0)
    
    return images, img_ids, orig_sizes

def analyze_dataset():
    """分析数据集，绘制类别分布等统计图表"""
    # 统计各类别的样本数量
    class_counts = {cls: 0 for cls in VOC_CLASSES}
    
    # 处理训练集
    anno_dir = TRAIN_DATA_PATH / "Annotations"
    for xml_file in tqdm(list(anno_dir.glob("*.xml")), desc="Analyzing dataset"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in class_counts:
                class_counts[name] += 1
    
    # 绘制类别分布条形图
    plt.figure(figsize=(14, 7))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('类别', fontproperties='SimHei', fontsize=12)
    plt.ylabel('样本数量', fontproperties='SimHei', fontsize=12)
    plt.title('VOC2007训练集中各类别样本分布', fontproperties='SimHei', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(VISUALIZATION_PATH / "class_distribution.png"))
    
    # 分析边界框大小分布
    box_areas = []
    box_aspects = []
    
    for xml_file in tqdm(list(anno_dir.glob("*.xml")), desc="Analyzing bounding boxes"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        width = float(root.find('size/width').text)
        height = float(root.find('size/height').text)
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 计算相对面积和宽高比
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            area = box_width * box_height
            aspect = box_width / box_height if box_height > 0 else 0
            
            box_areas.append(area)
            box_aspects.append(aspect)
    
    # 绘制边界框面积直方图
    plt.figure(figsize=(10, 6))
    plt.hist(box_areas, bins=50, alpha=0.75, color='blue')
    plt.xlabel('相对边界框面积', fontproperties='SimHei', fontsize=12)
    plt.ylabel('数量', fontproperties='SimHei', fontsize=12)
    plt.title('边界框相对面积分布', fontproperties='SimHei', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(VISUALIZATION_PATH / "bbox_area_distribution.png"))
    
    # 绘制边界框宽高比直方图
    plt.figure(figsize=(10, 6))
    plt.hist(box_aspects, bins=50, alpha=0.75, color='green')
    plt.xlabel('边界框宽高比', fontproperties='SimHei', fontsize=12)
    plt.ylabel('数量', fontproperties='SimHei', fontsize=12)
    plt.title('边界框宽高比分布', fontproperties='SimHei', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(VISUALIZATION_PATH / "bbox_aspect_distribution.png"))
    
    print(f"分析完成，图表已保存到 {VISUALIZATION_PATH}")

if __name__ == "__main__":
    # 分析数据集并生成可视化图表
    analyze_dataset()
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = create_data_loaders()
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")