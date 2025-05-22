import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'

# VOC数据集类别
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(Dataset):
    """
    PASCAL VOC数据集加载类
    """
    def __init__(self, root_dir, image_set='train', transform=None, target_transform=None, 
                 keep_difficult=False, input_size=512):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self.input_size = input_size
        
        self.image_set = image_set
        self._annopath = os.path.join(root_dir, 'VOCdevkit', 'VOC2007', 'Annotations', '%s.xml')
        self._imgpath = os.path.join(root_dir, 'VOCdevkit', 'VOC2007', 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(root_dir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', '%s.txt')
        
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip() for x in self.ids]
        
        self.class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # 加载图像
        img = Image.open(self._imgpath % img_id).convert("RGB")
        width, height = img.size
        
        # 加载标注
        target = self.parse_voc_xml(ET.parse(self._annopath % img_id).getroot())
        
        # 提取边界框和标签
        boxes = []
        labels = []
        difficulties = []
        
        for obj in target['annotation']['object']:
            difficult = int(obj['difficult']) if 'difficult' in obj else 0
            if not self.keep_difficult and difficult:
                continue
                
            bbox = obj['bndbox']
            bbox = [float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])]
            
            # 归一化边界框坐标
            bbox[0] /= width
            bbox[1] /= height
            bbox[2] /= width
            bbox[3] /= height
            
            boxes.append(bbox)
            labels.append(self.class_to_idx[obj['name']])
            difficulties.append(difficult)
        
        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        difficulties = torch.as_tensor(difficulties, dtype=torch.uint8)
        
        # 准备样本
        sample = {
            'image': img,
            'boxes': boxes,
            'labels': labels,
            'difficulties': difficulties,
            'image_id': img_id
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    @staticmethod
    def parse_voc_xml(node):
        """
        解析VOC XML格式标注文件
        """
        voc_dict = {}
        for child in node:
            if child.tag == 'object':
                if 'object' not in voc_dict:
                    voc_dict['object'] = []
                obj = {}
                for item in child:
                    if item.tag == 'bndbox':
                        bbox = {}
                        for box_item in item:
                            bbox[box_item.tag] = box_item.text
                        obj['bndbox'] = bbox
                    else:
                        obj[item.tag] = item.text
                voc_dict['object'].append(obj)
            else:
                voc_dict[child.tag] = VOCDataset.parse_voc_xml(child) if len(child) > 0 else child.text
        return voc_dict

class Compose:
    """
    组合多个变换操作
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor:
    """
    将PIL图像转换为张量
    """
    def __call__(self, sample):
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
        
        # 转换PIL图像为张量
        image = transforms.functional.to_tensor(image)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'difficulties': sample['difficulties'],
            'image_id': sample['image_id']
        }

class Resize:
    """
    调整图像大小
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes'].clone()
        
        # 获取当前尺寸
        old_width, old_height = image.size
        
        # 调整图像大小
        image = image.resize((self.size, self.size))
        
        # 已归一化的边界框无需调整
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': sample['labels'],
            'difficulties': sample['difficulties'],
            'image_id': sample['image_id']
        }

class RandomHorizontalFlip:
    """
    随机水平翻转
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes'].clone()
        
        if np.random.random() < self.p:
            # 翻转图像
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 翻转边界框
            boxes[:, 0], boxes[:, 2] = 1 - boxes[:, 2], 1 - boxes[:, 0]
            
        return {
            'image': image,
            'boxes': boxes,
            'labels': sample['labels'],
            'difficulties': sample['difficulties'],
            'image_id': sample['image_id']
        }

class ColorJitter:
    """
    色彩抖动增强
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, sample):
        image = sample['image']
        image = self.color_jitter(image)
        
        return {
            'image': image,
            'boxes': sample['boxes'],
            'labels': sample['labels'],
            'difficulties': sample['difficulties'],
            'image_id': sample['image_id']
        }

def collate_fn(batch):
    """
    自定义收集函数，处理变长数据
    """
    images = []
    boxes = []
    labels = []
    difficulties = []
    image_ids = []
    
    for b in batch:
        images.append(b['image'])
        boxes.append(b['boxes'])
        labels.append(b['labels'])
        difficulties.append(b['difficulties'])
        image_ids.append(b['image_id'])
        
    images = torch.stack(images, dim=0)
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'difficulties': difficulties,
        'image_ids': image_ids
    }

def get_data_loaders(data_dir, batch_size=16, input_size=512, num_workers=4):
    """
    创建训练、验证和测试数据加载器
    """
    # 定义变换
    train_transform = Compose([
        Resize(input_size),
        RandomHorizontalFlip(),
        ColorJitter(),
        ToTensor()
    ])
    
    test_transform = Compose([
        Resize(input_size),
        ToTensor()
    ])
    
    # 创建数据集
    train_dataset = VOCDataset(
        os.path.join(data_dir, 'VOCtrainval_06-Nov-2007'),
        image_set='train',
        transform=train_transform
    )
    
    val_dataset = VOCDataset(
        os.path.join(data_dir, 'VOCtrainval_06-Nov-2007'),
        image_set='val',
        transform=test_transform
    )
    
    test_dataset = VOCDataset(
        os.path.join(data_dir, 'VOCtest_06-Nov-2007'),
        image_set='test',
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def visualize_sample(sample, class_names=VOC_CLASSES):
    """
    可视化一个数据集样本
    """
    image = sample['image']
    boxes = sample['boxes']
    labels = sample['labels']
    
    # 转换张量为numpy进行可视化
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # 显示图像
    ax.imshow(image)
    
    # 获取图像尺寸
    height, width = image.shape[0], image.shape[1]
    
    # 绘制边界框
    for box, label in zip(boxes, labels):
        # 如果坐标是归一化的，转换为像素坐标
        if box[0] <= 1.0 and box[1] <= 1.0 and box[2] <= 1.0 and box[3] <= 1.0:
            x_min = box[0] * width
            y_min = box[1] * height
            x_max = box[2] * width
            y_max = box[3] * height
        else:
            x_min, y_min, x_max, y_max = box
            
        # 转换张量为int
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        # 获取类别名称
        class_name = class_names[label]
        
        # 创建矩形框
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(x_min, y_min - 5, f"{class_name}", 
                color='white', bbox=dict(facecolor='red', alpha=0.8))
    
    plt.axis('off')
    return fig

def plot_class_distribution(dataset, output_path, class_names=VOC_CLASSES):
    """
    绘制数据集中各类别的分布
    """
    class_counts = {cls: 0 for cls in class_names}
    
    for i in range(len(dataset)):
        sample = dataset[i]
        for label in sample['labels']:
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[class_names[label]] += 1
    
    # 按数量排序
    sorted_counts = {k: v for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)}
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(sorted_counts.keys(), sorted_counts.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('类别分布', fontsize=16)
    plt.xlabel('类别', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.tight_layout()
    
    # 在柱状图上添加数量标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def visualize_augmentations(dataset, output_path, num_samples=5):
    """
    可视化数据增强效果
    """
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(20, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        for j in range(3):  # 生成3个增强版本
            plt.subplot(num_samples, 4, i * 4 + j + 1)
            
            # 获取新的增强样本
            if j > 0:
                sample = dataset[idx]
                
            # 可视化
            img = sample['image']
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                
            plt.imshow(img)
            plt.title(f"样本 {idx} - 增强版本 {j}")
            plt.axis('off')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 测试代码
if __name__ == "__main__":
    data_dir = "../data"
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=2)
    
    # 获取一个样本
    for batch in train_loader:
        images = batch['images']
        boxes = batch['boxes']
        labels = batch['labels']
        print(f"批次形状: {images.shape}")
        print(f"边界框数量: {[len(b) for b in boxes]}")
        break