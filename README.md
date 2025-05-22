# 目标检测实验 - 基于 Mamba 技术的 PASCAL VOC 2007 数据集实现

## 数据集分析

您提供的 PASCAL VOC 2007 数据集是目标检测领域的经典数据集，包含训练验证集和测试集，结构完整规范：

- `Annotations`: XML 格式的标注文件，包含边界框坐标和类别信息
- `ImageSets`: 训练/验证/测试图像的索引信息
- `JPEGImages`: 原始图像文件
- `SegmentationClass`/`SegmentationObject`: 分割任务相关的标注
- 数据集包含 20 个常见对象类别（如人、车、动物等）

## 实验流程

以下是完整的实验流程和代码文件安排：

### 代码文件结构

```
mamba_code/
├── data_processing.py   # 数据处理与加载
├── model.py             # Mamba增强型目标检测模型
├── utils.py             # 工具函数(评估、可视化等)
├── train.py             # 训练脚本
├── test.py              # 测试脚本
└── results/             # 实验结果输出目录
    ├── models/          # 保存的模型
    ├── plots/           # 训练过程可视化
    ├── dataset/         # 数据集统计可视化
    └── test/            # 测试结果
        ├── predictions/ # 预测可视化
        └── metrics/     # 测试指标
```

### 实验步骤与技术选择

1. **数据预处理**
   - 解析 VOC XML 标注格式
   - 数据增强：随机翻转、色彩抖动等
   - 标准化、张量化

2. **模型架构**
   - 基础特征提取：ResNet50
   - 特征增强：Mamba 序列建模层
   - 检测头：Faster R-CNN

3. **训练策略**
   - SGD 优化器 + 学习率调度
   - 多尺度训练
   - 批处理策略

4. **评估与可视化**
   - mAP 计算
   - 精确率-召回率曲线
   - 混淆矩阵
   - 检测结果可视化

## 代码实现

### 1. 数据处理 (mamba_code/data_processing.py)

```python
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
```

### 2. 模型定义 (mamba_code/model.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection import FasterRCNN
from mamba_ssm.models.mixer_seq_simple import MambaBlock

class MambaFeatureExtractor(nn.Module):
    """Mamba特征提取器: 将特征图按序列处理"""
    def __init__(self, in_channels, d_model=512, depth=2):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        
        # 初始投影
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Mamba层
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, 
                      d_state=16,
                      d_conv=4,
                      expand=2)
            for _ in range(depth)
        ])
        
        # 最终投影回原始通道深度
        self.output_proj = nn.Conv2d(d_model, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # 初始投影
        x = self.input_proj(x)
        
        # 重塑以进行序列建模: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.flatten(2).permute(0, 2, 1)
        
        # 应用Mamba块
        for block in self.mamba_blocks:
            x_seq = block(x_seq)
        
        # 重塑回原始形状: (B, H*W, C) -> (B, C, H, W)
        x = x_seq.permute(0, 2, 1).view(batch, self.d_model, height, width)
        
        # 最终投影
        x = self.output_proj(x)
        
        return x

class MambaBackbone(nn.Module):
    """带Mamba增强的ResNet骨干网络"""
    def __init__(self, backbone_name='resnet50', pretrained=True, trainable_layers=3):
        super().__init__()
        
        # 获取CNN骨干网络
        if backbone_name == 'resnet50':
            # 使用预训练的ResNet50作为基础CNN特征提取器
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            # 提取各阶段
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1  # 256通道
            self.layer2 = backbone.layer2  # 512通道
            self.layer3 = backbone.layer3  # 1024通道
            self.layer4 = backbone.layer4  # 2048通道
            
            # 特征通道数
            self.out_channels = 2048
            
            # 将Mamba增强应用于选定层
            self.mamba_layer3 = MambaFeatureExtractor(in_channels=1024, d_model=512, depth=2)
            self.mamba_layer4 = MambaFeatureExtractor(in_channels=2048, d_model=512, depth=2)
            
            # 冻结一些层
            self._freeze_layers(trainable_layers)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
            
    def _freeze_layers(self, trainable_layers):
        """冻结选定层以进行迁移学习"""
        # 冻结特定层
        layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        
        # 首先冻结所有层
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
                
        # 解冻指定数量的可训练层
        for layer in layers[-trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # 对layer3应用Mamba
        x = self.layer3(x)
        x = x + self.mamba_layer3(x)  # 添加残差连接
        
        # 对layer4应用Mamba
        x = self.layer4(x)
        x = x + self.mamba_layer4(x)  # 添加残差连接
        
        return x

class MambaRCNN(nn.Module):
    """带Mamba增强骨干网络的Faster R-CNN"""
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True):
        super().__init__()
        
        # 初始化骨干网络
        self.backbone = MambaBackbone(backbone_name=backbone_name, pretrained=pretrained)
        
        # 创建锚点生成器
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # 创建ROI池化层
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # 创建Faster R-CNN模型
        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=600,
            max_size=1000,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

# 创建模型的工具函数
def create_model(num_classes=21, pretrained=True):  # 20个类 + 背景
    model = MambaRCNN(num_classes=num_classes, pretrained=pretrained)
    return model

# 测试代码
if __name__ == "__main__":
    # 测试模型
    model = create_model()
    x = torch.randn(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print("模型输出键:", output[0].keys())
```

### 3. 工具函数 (mamba_code/utils.py)

```python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
from PIL import Image
import torchvision.transforms.functional as F

def box_iou(box1, box2):
    """
    计算边界框的IoU
    边界框格式: [x1, y1, x2, y2]
    """
    # 计算交集面积
    x1 = np.maximum(box1[:, 0][:, np.newaxis], box2[:, 0])
    y1 = np.maximum(box1[:, 1][:, np.newaxis], box2[:, 1])
    x2 = np.minimum(box1[:, 2][:, np.newaxis], box2[:, 2])
    y2 = np.minimum(box1[:, 3][:, np.newaxis], box2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 计算面积
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # 计算并集面积
    union = box1_area[:, np.newaxis] + box2_area - intersection
    
    # 计算IoU
    iou = intersection / union
    
    return iou

def calculate_ap(recall, precision):
    """
    使用11点插值法计算平均精度
    """
    # 11点插值
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap

def calculate_map(all_detections, all_targets, iou_threshold=0.5, class_names=None):
    """
    计算目标检测的mAP
    
    参数:
        all_detections: 每张图像的检测结果列表
        all_targets: 每张图像的真实目标列表
        iou_threshold: 真阳性的IoU阈值
        class_names: 类别名称列表
        
    返回:
        包含mAP和每类AP值的字典
    """
    # 初始化变量
    aps = {}
    pr_curves = {}
    
    # 处理每个类别
    for class_idx, class_name in enumerate(class_names):
        # 提取此类别的检测结果和真实标签
        class_detections = []
        class_targets = []
        
        for img_dets, img_gts in zip(all_detections, all_targets):
            # 获取此类别的检测结果
            idx = np.where(img_dets['labels'] == class_idx)[0]
            boxes = img_dets['boxes'][idx]
            scores = img_dets['scores'][idx]
            
            # 存储检测结果 (box, score)
            for box, score in zip(boxes, scores):
                class_detections.append({
                    'box': box,
                    'score': score,
                    'image_id': img_dets['image_id']
                })
            
            # 获取此类别的真实标签
            idx = np.where(img_gts['labels'] == class_idx)[0]
            gt_boxes = img_gts['boxes'][idx]
            
            # 存储真实标签
            class_targets.append({
                'boxes': gt_boxes,
                'image_id': img_gts['image_id'],
                'detected': [False] * len(gt_boxes)  # 跟踪已检测到的真实框
            })
        
        # 按置信度降序排序检测结果
        class_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 初始化真阳性和假阳性数组
        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        
        # 将检测结果分配给真实目标
        for det_idx, detection in enumerate(class_detections):
            # 查找同一图像的真实标签
            gt = None
            for t in class_targets:
                if t['image_id'] == detection['image_id']:
                    gt = t
                    break
            
            # 如果没有真实标签，则为假阳性
            if gt is None or len(gt['boxes']) == 0:
                fp[det_idx] = 1
                continue
            
            # 计算与所有真实框的IoU
            ious = box_iou(np.array([detection['box']]), gt['boxes'])
            max_iou = np.max(ious)
            max_iou_idx = np.argmax(ious)
            
            # 检查是否为真阳性
            if max_iou >= iou_threshold and not gt['detected'][max_iou_idx]:
                tp[det_idx] = 1
                gt['detected'][max_iou_idx] = True
            else:
                fp[det_idx] = 1
        
        # 计算精确率和召回率
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        
        # 检查是否有检测结果
        if len(tp_cumsum) > 0:
            recall = tp_cumsum / sum(len(gt['boxes']) for gt in class_targets)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        else:
            recall = np.array([0])
            precision = np.array([0])
        
        # 确保精确率从1.0开始
        precision = np.concatenate(([1], precision))
        recall = np.concatenate(([0], recall))
        
        # 计算AP
        ap = calculate_ap(recall, precision)
        aps[class_name] = ap
        
        # 存储精确率-召回率曲线
        pr_curves[class_name] = {
            'precision': precision,
            'recall': recall
        }
    
    # 计算mAP
    mAP = np.mean(list(aps.values()))
    
    return {
        'mAP': mAP,
        'per_class_ap': aps,
        'per_class_pr': pr_curves
    }

def visualize_predictions(image, target, prediction, class_names=None, title=None):
    """
    可视化带有真实和预测边界框的图像
    
    参数:
        image: 张量或PIL图像
        target: 包含'boxes'和'labels'的字典
        prediction: 包含'boxes'、'labels'和'scores'的字典
        class_names: 类别名称列表
        title: 可选的图表标题
        
    返回:
        matplotlib图形
    """
    # 转换张量为numpy进行可视化
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        
        # 如果需要，反归一化
        image = (image * 255).astype(np.uint8)
    
    # 创建图形和轴
    fig, ax = plt.subplots(1, figsize=(12, 12))
    
    # 显示图像
    ax.imshow(image)
    
    # 获取图像尺寸
    height, width = image.shape[0], image.shape[1]
    
    # 绘制真实边界框
    if target is not None and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            # 如果坐标是归一化的，转换为像素坐标
            if isinstance(box, torch.Tensor):
                box = box.numpy()
                
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
            if class_names is not None:
                class_name = class_names[label]
            else:
                class_name = f"类别 {label}"
                
            # 创建矩形框
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x_min, y_min - 5, f"真实: {class_name}", 
                    color='white', bbox=dict(facecolor='green', alpha=0.8))
    
    # 绘制预测边界框
    if prediction is not None and len(prediction['boxes']) > 0:
        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            # 如果坐标是归一化的，转换为像素坐标
            if isinstance(box, torch.Tensor):
                box = box.numpy()
                
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
            if isinstance(score, torch.Tensor):
                score = score.item()
                
            # 获取类别名称
            if class_names is not None:
                class_name = class_names[label]
            else:
                class_name = f"类别 {label}"
                
            # 创建矩形框
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x_min, y_max + 15, f"预测: {class_name} ({score:.2f})", 
                    color='white', bbox=dict(facecolor='red', alpha=0.8))
    
    # 设置标题
    if title:
        plt.title(title, fontsize=16)
        
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def calculate_confusion_matrix(pred_classes, true_classes, num_classes):
    """
    计算混淆矩阵
    
    参数:
        pred_classes: 预测类别列表
        true_classes: 真实类别列表
        num_classes: 类别总数
        
    返回:
        混淆矩阵
    """
    # 转换为numpy数组
    if isinstance(pred_classes, torch.Tensor):
        pred_classes = pred_classes.cpu().numpy()
    if isinstance(true_classes, torch.Tensor):
        true_classes = true_classes.cpu().numpy()
        
    pred_classes = np.array(pred_classes)
    true_classes = np.array(true_classes)
    
    # 初始化混淆矩阵
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # 填充混淆矩阵
    for i in range(len(true_classes)):
        if i < len(pred_classes):
            conf_matrix[true_classes[i], pred_classes[i]] += 1
    
    return conf_matrix
```

### 4. 训练脚本 (mamba_code/train.py)

```python
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
from tqdm import tqdm
import json

from model import create_model
from data_processing import get_data_loaders, VOC_CLASSES, plot_class_distribution, visualize_augmentations
from utils import calculate_map, visualize_predictions

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    训练一个epoch
    """
    model.train()
    
    # 初始化指标
    loss_classifier_sum = 0
    loss_box_reg_sum = 0
    loss_objectness_sum = 0
    loss_rpn_box_reg_sum = 0
    total_loss_sum = 0
    
    # 进度条
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for i, batch in enumerate(pbar):
        images = batch['images'].to(device)
        
        targets = []
        for j in range(len(batch['boxes'])):
            target = {
                'boxes': batch['boxes'][j].to(device),
                'labels': batch['labels'][j].to(device)
            }
            targets.append(target)
        
        # 前向传播
        loss_dict = model(images, targets)
        
        # 计算总损失
        losses = sum(loss for loss in loss_dict.values())
        
        # 更新指标
        loss_classifier_sum += loss_dict['loss_classifier'].item()
        loss_box_reg_sum += loss_dict['loss_box_reg'].item()
        loss_objectness_sum += loss_dict['loss_objectness'].item()
        loss_rpn_box_reg_sum += loss_dict['loss_rpn_box_reg'].item()
        total_loss_sum += losses.item()
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'cls_loss': f"{loss_dict['loss_classifier'].item():.4f}",
            'box_loss': f"{loss_dict['loss_box_reg'].item():.4f}"
        })
    
    # 计算平均损失
    num_batches = len(data_loader)
    avg_loss_classifier = loss_classifier_sum / num_batches
    avg_loss_box_reg = loss_box_reg_sum / num_batches
    avg_loss_objectness = loss_objectness_sum / num_batches
    avg_loss_rpn_box_reg = loss_rpn_box_reg_sum / num_batches
    avg_total_loss = total_loss_sum / num_batches
    
    # 返回指标
    metrics = {
        'loss_classifier': avg_loss_classifier,
        'loss_box_reg': avg_loss_box_reg,
        'loss_objectness': avg_loss_objectness,
        'loss_rpn_box_reg': avg_loss_rpn_box_reg,
        'total_loss': avg_total_loss
    }
    
    return metrics

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, output_dir, class_names=VOC_CLASSES):
    """
    在验证集上评估模型
    """
    model.eval()
    
    # 初始化指标
    all_detections = []
    all_targets = []
    
    # 进度条
    pbar = tqdm(data_loader, desc=f"验证 {epoch}")
    
    for i, batch in enumerate(pbar):
        images = batch['images'].to(device)
        
        # 存储真实标签
        for j in range(len(batch['boxes'])):
            img_gt = {
                'boxes': batch['boxes'][j].cpu().numpy(),
                'labels': batch['labels'][j].cpu().numpy(),
                'image_id': batch['image_ids'][j]
            }
            all_targets.append(img_gt)
        
        # 获取预测结果
        outputs = model(images)
        
        # 存储预测结果
        for j, output in enumerate(outputs):
            img_pred = {
                'boxes': output['boxes'].cpu().numpy(),
                'scores': output['scores'].cpu().numpy(),
                'labels': output['labels'].cpu().numpy(),
                'image_id': batch['image_ids'][j]
            }
            all_detections.append(img_pred)
        
        # 可视化当前epoch的一些预测结果
        if i == 0 and epoch % 5 == 0:
            for j in range(min(2, len(images))):
                img = images[j].cpu()
                target = {
                    'boxes': batch['boxes'][j],
                    'labels': batch['labels'][j]
                }
                pred = {
                    'boxes': outputs[j]['boxes'].cpu(),
                    'labels': outputs[j]['labels'].cpu(),
                    'scores': outputs[j]['scores'].cpu()
                }
                
                # 只使用置信度 > 0.5 的预测
                mask = pred['scores'] > 0.5
                pred['boxes'] = pred['boxes'][mask]
                pred['labels'] = pred['labels'][mask]
                pred['scores'] = pred['scores'][mask]
                
                fig = visualize_predictions(
                    img, target, pred, 
                    class_names=class_names,
                    title=f"Epoch {epoch} - 样本 {j}"
                )
                
                # 保存可视化结果
                os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
                fig.savefig(os.path.join(output_dir, 'predictions', f'epoch_{epoch}_sample_{j}.png'))
                plt.close(fig)
    
    # 计算mAP
    metrics = calculate_map(all_detections, all_targets, class_names=class_names)
    
    return metrics

def plot_metrics(train_metrics, val_metrics, output_dir):
    """绘制并保存训练和验证指标"""
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取轮次数
    epochs = list(range(1, len(train_metrics['total_loss']) + 1))
    
    # 绘制训练损失
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_metrics['loss_classifier'], label='分类损失')
    plt.plot(epochs, train_metrics['loss_box_reg'], label='边界框回归损失')
    plt.plot(epochs, train_metrics['loss_objectness'], label='物体性损失')
    plt.plot(epochs, train_metrics['loss_rpn_box_reg'], label='RPN边界框回归损失')
    plt.plot(epochs, train_metrics['total_loss'], label='总损失', linewidth=2)
    plt.title('训练过程损失曲线', fontsize=16)
    plt.xlabel('轮次 (Epoch)', fontsize=14)
    plt.ylabel('损失值 (Loss)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_losses.png'))
    plt.close()
    
    # 绘制验证mAP
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_metrics['mAP'], label='平均精度 (mAP)', marker='o', linewidth=2)
    plt.title('验证集平均精度 (mAP)', fontsize=16)
    plt.xlabel('轮次 (Epoch)', fontsize=14)
    plt.ylabel('mAP', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_map.png'))
    plt.close()
    
    # 绘制最后一轮的每类AP
    plt.figure(figsize=(14, 10))
    classes = list(val_metrics['per_class_ap'][-1].keys())
    ap_values = list(val_metrics['per_class_ap'][-1].values())
    
    # 按AP值排序
    sorted_indices = np.argsort(ap_values)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_ap_values = [ap_values[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_classes, sorted_ap_values, color='skyblue')
    plt.title('各类别的平均精度 (AP)', fontsize=16)
    plt.xlabel('类别', fontsize=14)
    plt.ylabel('AP', fontsize=14)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # 在柱状图上添加值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_ap.png'))
    plt.close()
    
    # 绘制选定类别的精确率-召回率曲线（最后一轮）
    plt.figure(figsize=(12, 8))
    
    # 选择几个重要类别
    selected_classes = ['person', 'car', 'dog', 'chair', 'bottle']
    for cls in selected_classes:
        if cls in val_metrics['per_class_pr'][-1]:
            precision = val_metrics['per_class_pr'][-1][cls]['precision']
            recall = val_metrics['per_class_pr'][-1][cls]['recall']
            plt.plot(recall, precision, label=f'{cls} (AP: {val_metrics["per_class_ap"][-1][cls]:.2f})', linewidth=2)
    
    plt.title('精确率-召回率曲线', fontsize=16)
    plt.xlabel('召回率 (Recall)', fontsize=14)
    plt.ylabel('精确率 (Precision)', fontsize=14)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
    plt.close()

def train_model(data_dir, output_dir='mamba_code/results', num_epochs=20, 
               batch_size=8, learning_rate=0.001, device='cuda'):
    """主训练函数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size=batch_size)
    
    # 可视化数据集统计
    print("生成数据集可视化...")
    plot_class_distribution(train_loader.dataset, 
                           os.path.join(output_dir, 'dataset', 'train_class_distribution.png'))
    plot_class_distribution(val_loader.dataset, 
                           os.path.join(output_dir, 'dataset', 'val_class_distribution.png'))
    visualize_augmentations(train_loader.dataset, 
                           os.path.join(output_dir, 'dataset', 'data_augmentations.png'))
    
    # 创建模型
    print("创建模型...")
    model = create_model(num_classes=len(VOC_CLASSES) + 1)  # +1用于背景
    model.to(device)
    
    # 初始化优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 初始化TensorBoard写入器
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # 初始化指标存储
    train_metrics = {
        'loss_classifier': [],
        'loss_box_reg': [],
        'loss_objectness': [],
        'loss_rpn_box_reg': [],
        'total_loss': []
    }
    
    val_metrics = {
        'mAP': [],
        'per_class_ap': [],
        'per_class_pr': []
    }
    
    # 开始计时
    start_time = time.time()
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\n第 {epoch}/{num_epochs} 轮")
        
        # 训练一个epoch
        epoch_train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 在验证集上评估
        epoch_val_metrics = evaluate(model, val_loader, device, epoch, output_dir)
        
        # 存储指标
        for k, v in epoch_train_metrics.items():
            train_metrics[k].append(v)
            writer.add_scalar(f'train/{k}', v, epoch)
        
        val_metrics['mAP'].append(epoch_val_metrics['mAP'])
        val_metrics['per_class_ap'].append(epoch_val_metrics['per_class_ap'])
        val_metrics['per_class_pr'].append(epoch_val_metrics['per_class_pr'])
        writer.add_scalar('val/mAP', epoch_val_metrics['mAP'], epoch)
        
        # 将每类AP添加到TensorBoard
        for cls, ap in epoch_val_metrics['per_class_ap'].items():
            writer.add_scalar(f'val/AP_{cls}', ap, epoch)
        
        # 打印当前指标
        print(f"训练损失: {epoch_train_metrics['total_loss']:.4f}, "
              f"验证mAP: {epoch_val_metrics['mAP']:.4f}")
        
        # 保存模型检查点
        if epoch % 5 == 0 or epoch == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(model_dir, f'model_epoch_{epoch}.pth'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_final.pth'))
    
    # 绘制并保存指标
    plot_metrics(train_metrics, val_metrics, os.path.join(output_dir, 'plots'))
    
    # 将指标保存为JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        # 将numpy数组转换为列表以进行JSON序列化
        serializable_val_metrics = {
            'mAP': val_metrics['mAP'],
            'per_class_ap': [
                {cls: float(ap) for cls, ap in epoch_ap.items()}
                for epoch_ap in val_metrics['per_class_ap']
            ]
        }
        
        json.dump({
            'train': train_metrics,
            'val': serializable_val_metrics
        }, f, indent=4)
    
    # 计算训练时间
    total_time = time.time() - start_time
    print(f"训练完成，用时 {str(datetime.timedelta(seconds=int(total_time)))}")
    
    return model, train_metrics, val_metrics

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置数据目录
    data_dir = "../data"
    output_dir = "mamba_code/results"
    
    # 训练模型
    model, train_metrics, val_metrics = train_model(
        data_dir=data_dir,
        output_dir=output_dir,
        num_epochs=20,
        batch_size=4,
        learning_rate=0.001,
        device=device
    )
    
    print("训练成功完成!")
```

### 5. 测试脚本 (mamba_code/test.py)

```python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
from tqdm import tqdm
import json
import cv2

from model import create_model
from data_processing import get_data_loaders, VOC_CLASSES
from utils import calculate_map, visualize_predictions, calculate_confusion_matrix

def evaluate_model(model, data_loader, device, output_dir, class_names=VOC_CLASSES):
    """
    在测试集上评估模型并生成详细的指标和可视化
    """
    model.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    
    # 初始化指标
    all_detections = []
    all_targets = []
    
    # 存储所有预测和真实类别用于混淆矩阵
    all_pred_classes = []
    all_true_classes = []
    
    # 进度条
    pbar = tqdm(data_loader, desc="测试中")
    
    # 可视化一部分预测结果
    samples_to_visualize = np.random.choice(len(data_loader.dataset), 
                                          size=min(20, len(data_loader.dataset)), 
                                          replace=False)
    visualized_count = 0
    
    for i, batch in enumerate(pbar):
        images = batch['images'].to(device)
        
        # 存储真实标签
        for j in range(len(batch['boxes'])):
            img_gt = {
                'boxes': batch['boxes'][j].cpu().numpy(),
                'labels': batch['labels'][j].cpu().numpy(),
                'image_id': batch['image_ids'][j]
            }
            all_targets.append(img_gt)
            
            # 存储真实类别用于混淆矩阵
            all_true_classes.extend(batch['labels'][j].cpu().numpy())
        
        # 获取预测结果
        with torch.no_grad():
            outputs = model(images)
        
        # 存储预测结果
        for j, output in enumerate(outputs):
            img_pred = {
                'boxes': output['boxes'].cpu().numpy(),
                'scores': output['scores'].cpu().numpy(),
                'labels': output['labels'].cpu().numpy(),
                'image_id': batch['image_ids'][j]
            }
            all_detections.append(img_pred)
            
            # 存储预测类别用于混淆矩阵（置信度 > 0.5）
            mask = output['scores'].cpu().numpy() > 0.5
            all_pred_classes.extend(output['labels'].cpu().numpy()[mask])
            
            # 可视化选定样本
            if visualized_count < len(samples_to_visualize) and i * batch['images'].shape[0] + j in samples_to_visualize:
                img = images[j].cpu()
                target = {
                    'boxes': batch['boxes'][j],
                    'labels': batch['labels'][j]
                }
                pred = {
                    'boxes': output['boxes'].cpu(),
                    'labels': output['labels'].cpu(),
                    'scores': output['scores'].cpu()
                }
                
                # 只使用置信度 > 0.5 的预测
                mask = pred['scores'] > 0.5
                pred['boxes'] = pred['boxes'][mask]
                pred['labels'] = pred['labels'][mask]
                pred['scores'] = pred['scores'][mask]
                
                fig = visualize_predictions(
                    img, target, pred, 
                    class_names=class_names,
                    title=f"测试样本 {batch['image_ids'][j]}"
                )
                
                # 保存可视化
                fig.savefig(os.path.join(output_dir, 'predictions', f'test_sample_{batch["image_ids"][j]}.png'))
                plt.close(fig)
                
                visualized_count += 1
    
    # 计算mAP和其他指标
    metrics = calculate_map(all_detections, all_targets, class_names=class_names)
    
    # 计算混淆矩阵
    confusion_matrix = calculate_confusion_matrix(all_pred_classes, all_true_classes, num_classes=len(class_names)+1)
    
    # 保存指标
    with open(os.path.join(output_dir, 'metrics', 'test_metrics.json'), 'w') as f:
        # 将numpy值转换为原生Python类型以# filepath: mamba_code/test.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
from tqdm import tqdm
import json
import cv2

from model import create_model
from data_processing import get_data_loaders, VOC_CLASSES
from utils import calculate_map, visualize_predictions, calculate_confusion_matrix

def evaluate_model(model, data_loader, device, output_dir, class_names=VOC_CLASSES):
    """
    在测试集上评估模型并生成详细的指标和可视化
    """
    model.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    
    # 初始化指标
    all_detections = []
    all_targets = []
    
    # 存储所有预测和真实类别用于混淆矩阵
    all_pred_classes = []
    all_true_classes = []
    
    # 进度条
    pbar = tqdm(data_loader, desc="测试中")
    
    # 可视化一部分预测结果
    samples_to_visualize = np.random.choice(len(data_loader.dataset), 
                                          size=min(20, len(data_loader.dataset)), 
                                          replace=False)
    visualized_count = 0
    
    for i, batch in enumerate(pbar):
        images = batch['images'].to(device)
        
        # 存储真实标签
        for j in range(len(batch['boxes'])):
            img_gt = {
                'boxes': batch['boxes'][j].cpu().numpy(),
                'labels': batch['labels'][j].cpu().numpy(),
                'image_id': batch['image_ids'][j]
            }
            all_targets.append(img_gt)
            
            # 存储真实类别用于混淆矩阵
            all_true_classes.extend(batch['labels'][j].cpu().numpy())
        
        # 获取预测结果
        with torch.no_grad():
            outputs = model(images)
        
        # 存储预测结果
        for j, output in enumerate(outputs):
            img_pred = {
                'boxes': output['boxes'].cpu().numpy(),
                'scores': output['scores'].cpu().numpy(),
                'labels': output['labels'].cpu().numpy(),
                'image_id': batch['image_ids'][j]
            }
            all_detections.append(img_pred)
            
            # 存储预测类别用于混淆矩阵（置信度 > 0.5）
            mask = output['scores'].cpu().numpy() > 0.5
            all_pred_classes.extend(output['labels'].cpu().numpy()[mask])
            
            # 可视化选定样本
            if visualized_count < len(samples_to_visualize) and i * batch['images'].shape[0] + j in samples_to_visualize:
                img = images[j].cpu()
                target = {
                    'boxes': batch['boxes'][j],
                    'labels': batch['labels'][j]
                }
                pred = {
                    'boxes': output['boxes'].cpu(),
                    'labels': output['labels'].cpu(),
                    'scores': output['scores'].cpu()
                }
                
                # 只使用置信度 > 0.5 的预测
                mask = pred['scores'] > 0.5
                pred['boxes'] = pred['boxes'][mask]
                pred['labels'] = pred['labels'][mask]
                pred['scores'] = pred['scores'][mask]
                
                fig = visualize_predictions(
                    img, target, pred, 
                    class_names=class_names,
                    title=f"测试样本 {batch['image_ids'][j]}"
                )
                
                # 保存可视化
                fig.savefig(os.path.join(output_dir, 'predictions', f'test_sample_{batch["image_ids"][j]}.png'))
                plt.close(fig)
                
                visualized_count += 1
    
    # 计算mAP和其他指标
    metrics = calculate_map(all_detections, all_targets, class_names=class_names)
    
    # 计算混淆矩阵
    confusion_matrix = calculate_confusion_matrix(all_pred_classes, all_true_classes, num_classes=len(class_names)+1)
    
    # 保存指标
    with open(os.path.join(output_dir, 'metrics', 'test_metrics.json'), 'w') as f:
        # 将numpy值转换为原生Python类型以

找到具有 1 个许可证类型的类似代码