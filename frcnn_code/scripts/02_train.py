import os
import yaml
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import transforms as T
from engine import train_with_progress, evaluate
import utils
import datetime
import time
import argparse
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2
import sys
import platform
import logging
import matplotlib
from matplotlib.patches import Rectangle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import box_iou
from torch.cuda.amp import GradScaler, autocast

# 添加到脚本开头，导入后立即设置
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations\.check_version")


# 添加当前目录到路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
        plt.style.use('ggplot')  # 使用更美观的样式
        return True
    except Exception as e:
        print(f"字体设置失败: {e}, 可能会导致中文显示异常")
        return False
    


class InsectsDataset(torch.utils.data.Dataset):
    def __init__(self, root, coco_annotation_file, transforms=None, is_train=False):
        self.root = root
        self.transforms = transforms
        self.is_train = is_train
        
        # 加载COCO格式的注释
        with open(coco_annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # 创建图像ID到注释的映射
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # 只保留有注释的图像
        self.images = []
        for img in self.coco_data['images']:
            if img['id'] in self.img_to_anns:
                self.images.append(img)
        
        # 创建类别映射
        self.cat_mapping = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}
        self.categories = self.coco_data['categories']
        
        # 类别统计 - 用于显示类别分布
        self.class_counts = {cat['name']: 0 for cat in self.categories}
        for img_id, anns in self.img_to_anns.items():
            for ann in anns:
                cat_id = ann['category_id']
                for cat in self.categories:
                    if cat['id'] == cat_id:
                        self.class_counts[cat['name']] += 1
                        break
        
        print(f"数据集包含 {len(self.images)} 张图像, {len(self.categories)} 个类别")
        print(f"类别统计: {self.class_counts}")
    
    def get_class_weights(self):
        """计算类别权重（用于处理类别不平衡）"""
        counts = np.array([self.class_counts[cat['name']] for cat in self.categories])
        total = counts.sum()
        weights = total / (len(counts) * counts)
        weights = weights / weights.sum()  # 归一化
        return torch.tensor(weights, dtype=torch.float32)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        img_width, img_height = img.size
        
        # 获取该图像的所有注释
        anns = self.img_to_anns.get(img_info['id'], [])
        
        # 构建目标字典
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO格式的bbox是[x, y, width, height]
            # 转换为[x_min, y_min, x_max, y_max]
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            labels.append(self.cat_mapping[ann['category_id']])
        
        # 如果没有框，添加一个虚拟框避免错误
        if len(boxes) == 0:
            boxes.append([0, 0, 1, 1])
            labels.append(0)  # 背景类
            
        # 转换为Torch张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([img_info['id']])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([img_height, img_width], dtype=torch.int64)
        }
        
        # 转换PIL图像为numpy数组，以便使用albumentations
        img_np = np.array(img)
        
        if self.transforms is not None:
            if isinstance(self.transforms, A.Compose):
                # 如果使用albumentations
                try:
                    transformed = self.transforms(
                        image=img_np,
                        bboxes=boxes.numpy().tolist(),
                        labels=labels.numpy().tolist()
                    )
                    img_np = transformed['image']
                    
                    if len(transformed['bboxes']) > 0:
                        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                        labels = torch.tensor(transformed['labels'], dtype=torch.int64)
                    else:
                        # 如果增强后没有框，添加虚拟框
                        boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
                        labels = torch.tensor([0], dtype=torch.int64)
                    
                    target['boxes'] = boxes
                    target['labels'] = labels
                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    target['area'] = area
                    target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
                    
                    # 转换numpy数组为PyTorch张量
                    img = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
                except Exception as e:
                    print(f"转换错误: {e}")
                    # 回退到原始图像和注释
                    img = torch.from_numpy(np.array(Image.open(img_path).convert("RGB")).transpose(2, 0, 1)).float() / 255.0
            else:
                # 如果使用PyTorch的转换
                img, target = self.transforms(img, target)
        else:
            # 没有转换，手动将PIL图像转换为张量
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        
        return img, target
    
    def __len__(self):
        return len(self.images)

def get_transform_v2(train):
    """使用Albumentations库创建增强转换"""
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RGBShift(p=0.2),
            A.MotionBlur(p=0.1),
            A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # [x_min, y_min, x_max, y_max]
            label_fields=['labels']
        ))
    else:
        transform = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # [x_min, y_min, x_max, y_max]
            label_fields=['labels']
        ))
    
    return transform

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_faster_rcnn_model(num_classes, pretrained=True, backbone='resnet50'):
    """获取不同骨干网络的Faster R-CNN模型"""
    if backbone == 'resnet50':
        # 标准的ResNet50骨干网络
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    elif backbone == 'resnet101':
        # 使用更深的ResNet101骨干网络
        model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=pretrained)
    elif backbone == 'mobilenet':
        # 使用轻量级MobileNetV3骨干网络
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的骨干网络: {backbone}")
    
    # 替换分类器头以匹配我们的类别数量
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def save_training_plot(results, output_dir, class_names=None):
    """保存训练和验证损失曲线"""
    plt.figure(figsize=(14, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(results['train_loss'], label='训练损失')
    if 'val_loss' in results and results['val_loss']:
        plt.plot(results['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制mAP曲线
    if 'val_map' in results and results['val_map']:
        plt.subplot(2, 2, 2)
        plt.plot(results['val_map'], label='验证mAP', marker='o')
        plt.title('验证mAP')
        plt.xlabel('周期')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制学习率曲线
    if 'learning_rate' in results and results['learning_rate']:
        plt.subplot(2, 2, 3)
        plt.plot(results['learning_rate'], label='学习率')
        plt.title('学习率变化')
        plt.xlabel('周期')
        plt.ylabel('学习率')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
    # 绘制类别AP曲线
    if class_names and 'class_ap' in results and results['class_ap']:
        plt.subplot(2, 2, 4)
        class_ap_data = results['class_ap']
        for i, class_name in enumerate(class_names[1:]):  # 跳过背景类
            ap_values = [epoch_data.get(i-1, 0) for epoch_data in class_ap_data]  # 类别ID从0开始
            plt.plot(ap_values, label=f'{class_name}')
        plt.title('各类别AP')
        plt.xlabel('周期')
        plt.ylabel('AP')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # 将图例放在图外
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()

def visualize_predictions(model, data_loader, device, class_names, output_dir, score_threshold=0.5, max_images=10):
    """可视化模型在部分验证图像上的预测结果，添加错误处理"""
    try:
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用不同颜色表示不同类别
        colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
        colors = (colors[:, :3] * 255).astype(np.uint8).tolist()
        
        with torch.no_grad():
            img_count = 0
            for images, targets in tqdm(data_loader, desc="生成预测可视化"):
                if img_count >= max_images:
                    break
                    
                images = list(img.to(device) for img in images)
                try:
                    outputs = model(images)
                except Exception as e:
                    print(f"模型预测失败: {e}")
                    continue
                
                for j, (img, target, output) in enumerate(zip(images, targets, outputs)):
                    if img_count >= max_images:
                        break
                    
                    try:
                        # 转回CPU并转换为numpy图像
                        img_np = img.cpu().permute(1, 2, 0).numpy()
                        # 反归一化，恢复原始图像
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = std * img_np + mean
                        img_np = np.clip(img_np, 0, 1) * 255
                        img_np = img_np.astype(np.uint8)
                        
                        # 创建三张对比图：原始图像，真实标签，预测结果
                        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                        
                        # 原始图像
                        axs[0].imshow(img_np)
                        axs[0].set_title('原始图像')
                        axs[0].axis('off')
                        
                        # 真实边界框
                        axs[1].imshow(img_np)
                        for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
                            box = box.astype(np.int32)
                            color = colors[label % len(colors)]
                            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                            linewidth=2, edgecolor=[c/255 for c in color], facecolor='none')
                            axs[1].add_patch(rect)
                            axs[1].text(box[0], box[1]-5, class_names[label], 
                                      fontsize=8, color='white', 
                                      bbox=dict(facecolor=[c/255 for c in color], alpha=0.7))
                        axs[1].set_title('真实标注')
                        axs[1].axis('off')
                        
                        # 预测边界框
                        axs[2].imshow(img_np)
                        for box, label, score in zip(output['boxes'].cpu().numpy(), 
                                                  output['labels'].cpu().numpy(), 
                                                  output['scores'].cpu().numpy()):
                            if score > score_threshold:
                                box = box.astype(np.int32)
                                color = colors[label % len(colors)]
                                rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                                linewidth=2, edgecolor=[c/255 for c in color], facecolor='none')
                                axs[2].add_patch(rect)
                                axs[2].text(box[0], box[1]-5, f'{class_names[label]} {score:.2f}', 
                                          fontsize=8, color='white', 
                                          bbox=dict(facecolor=[c/255 for c in color], alpha=0.7))
                        axs[2].set_title('预测结果')
                        axs[2].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'pred_{img_count}.png'), dpi=200)
                        plt.close(fig)
                        
                        img_count += 1
                    except Exception as e:
                        print(f"可视化预测结果时出错 (图像 {img_count}): {e}")
                        # 确保即使当前图像处理失败，也会递增计数器避免无限循环
                        img_count += 1
                        # 确保图形被关闭
                        try:
                            plt.close()
                        except:
                            pass
    except Exception as e:
        print(f"预测可视化过程中发生错误: {e}")
        # 确保图形被关闭
        try:
            plt.close()
        except:
            pass

def calculate_class_ap(model, data_loader, device, num_classes):
    """计算每个类别的AP值"""
    model.eval()
    class_aps = {}
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="计算每类AP"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()
                
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # 存储预测和目标
                all_predictions.append((pred_boxes, pred_scores, pred_labels))
                all_targets.append((gt_boxes, gt_labels))
    
    # 计算每个类别的AP
    for class_id in range(1, num_classes):  # 跳过背景类(0)
        predictions_for_class = []
        targets_for_class = []
        
        for (pred_boxes, pred_scores, pred_labels), (gt_boxes, gt_labels) in zip(all_predictions, all_targets):
            # 筛选该类别的预测和真实框
            class_pred_indices = pred_labels == class_id
            class_gt_indices = gt_labels == class_id
            
            predictions_for_class.append({
                'boxes': pred_boxes[class_pred_indices],
                'scores': pred_scores[class_pred_indices]
            })
            
            targets_for_class.append({
                'boxes': gt_boxes[class_gt_indices]
            })
        
        # 计算该类别的AP
        ap = calculate_ap_for_class(predictions_for_class, targets_for_class)
        class_aps[class_id-1] = ap  # 注意这里键是类别ID减1，因为我们跳过背景类
    
    return class_aps

def calculate_ap_for_class(predictions, targets, iou_threshold=0.5):
    """计算单个类别的AP"""
    if not any(len(t['boxes']) > 0 for t in targets):
        return 0.0  # 如果没有真实框，AP为0
    
    # 收集所有预测和真实框
    all_preds = []
    all_gt_matched = []
    
    for pred, gt in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # 计算IoU矩阵
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        
        # 根据分数对预测进行排序
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # 记录每个预测框的得分和是否匹配到GT
        for i, box in enumerate(pred_boxes):
            score = pred_scores[i]
            iou_values = iou_matrix[sorted_indices[i]]
            
            # 如果IoU超过阈值，则认为是匹配的
            if len(iou_values) > 0 and torch.max(iou_values) >= iou_threshold:
                all_preds.append((float(score), 1))  # 真阳性
                # 标记这个GT已匹配，避免重复匹配
                max_iou_idx = torch.argmax(iou_values).item()
                iou_matrix[:, max_iou_idx] = -1  
            else:
                all_preds.append((float(score), 0))  # 假阳性
        
        # 记录未匹配的GT框
        all_gt_matched.extend([0] * len(gt_boxes))
    
    # 如果没有预测或真实框，AP为0
    if not all_preds:
        return 0.0
    
    # 按分数排序
    all_preds.sort(key=lambda x: x[0], reverse=True)
    
    # 计算精确率和召回率
    precisions = []
    recalls = []
    tp_cumsum = 0
    fp_cumsum = 0
    total_gt = len(all_gt_matched)
    
    for _, matched in all_preds:
        if matched == 1:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt if total_gt > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # 计算AP - 使用11点插值法
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if not recalls or recalls[-1] < t:
            p = 0
        else:
            # 找到大于等于t的最大召回率对应的精确率
            p = max([precisions[i] for i, r in enumerate(recalls) if r >= t]) if recalls else 0
        ap += p / 11
    
    return ap

# 添加到02_train.py文件中

def plot_learning_rate_heatmap(learning_rates, output_dir):
    """绘制学习率变化热力图，添加错误处理"""
    try:
        plt.figure(figsize=(10, 6))
        # 将学习率转为二维数组以便绘制热力图
        lr_data = np.array(learning_rates).reshape(1, -1)
        
        # 创建热力图
        im = plt.imshow(lr_data, cmap='hot', aspect='auto')
        plt.colorbar(im, label='学习率值')
        
        # 添加X轴标签（周期数）
        plt.xticks(range(len(learning_rates)), [f'{i+1}' for i in range(len(learning_rates))])
        plt.xlabel('训练周期')
        plt.yticks([])  # 隐藏Y轴刻度
        plt.title('学习率热力图')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_heatmap.png'), dpi=300)
    except Exception as e:
        print(f"绘制学习率热力图时出错: {e}")
    finally:
        # 确保图形被关闭
        try:
            plt.close()
        except:
            pass
        
def plot_loss_components(results, output_dir):
    """绘制损失组件变化图，添加错误处理"""
    try:
        if 'loss_components' not in results or not results['loss_components']:
            print("没有损失组件数据可绘制")
            return
        
        components = results['loss_components'][0].keys()
        epochs = range(1, len(results['loss_components']) + 1)
        
        plt.figure(figsize=(12, 8))
        
        for component in components:
            try:
                values = [epoch_data.get(component, 0) for epoch_data in results['loss_components']]
                plt.plot(epochs, values, marker='o', label=component)
            except Exception as e:
                print(f"绘制损失组件 {component} 时出错: {e}")
                continue
        
        plt.title('损失组件变化')
        plt.xlabel('训练周期')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_components.png'), dpi=300)
    except Exception as e:
        print(f"绘制损失组件变化图时出错: {e}")
    finally:
        # 确保图形被关闭
        try:
            plt.close()
        except:
            pass

def plot_confusion_matrix(cm, class_names, output_dir, epoch=None):
    """绘制混淆矩阵，添加错误处理"""
    try:
        # 创建热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'混淆矩阵 {f"- Epoch {epoch}" if epoch is not None else ""}')
        plt.colorbar()
        
        # 添加刻度标签
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # 在每个单元格中添加数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                try:
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                except Exception as e:
                    print(f"添加混淆矩阵单元格文本时出错 ({i},{j}): {e}")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 保存图像
        file_name = f'confusion_matrix_epoch_{epoch}.png' if epoch is not None else 'confusion_matrix.png'
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")
    finally:
        # 确保图形被关闭
        try:
            plt.close()
        except:
            pass

def calculate_confusion_matrix(model, data_loader, device, num_classes, epoch=None):
    """计算混淆矩阵"""
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    model.eval()
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="计算混淆矩阵"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # 获取预测的类别和分数
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                # 只考虑得分高于阈值的预测
                keep = pred_scores > 0.5
                pred_labels = pred_labels[keep]
                
                # 如果没有预测或没有真实框，则跳过
                if len(pred_labels) == 0 or len(gt_labels) == 0:
                    continue
                    
                # 匹配每个真实框到预测框
                for gt_label in gt_labels:
                    # 如果有预测，则与最高得分的匹配
                    if len(pred_labels) > 0:
                        pred_label = pred_labels[0]  # 取第一个预测（假设已按分数排序）
                        confusion_matrix[gt_label, pred_label] += 1
                    else:
                        # 没有预测，算作未检测
                        confusion_matrix[gt_label, 0] += 1  # 假设0是背景类
    
    return confusion_matrix

def train_model(data_dir, output_dir, config_path, num_epochs=10, batch_size=2, lr=0.005, 
                backbone='resnet50', use_albumentations=True, use_amp=True, num_workers=4):
    """训练Faster R-CNN模型，支持混合精度训练"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 创建混淆矩阵存储目录
    os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)
    
    # 存储训练结果
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_map': [],
        'learning_rate': [],
        'class_ap': [],
        'loss_components': []  # 添加损失组件的跟踪
    }
    
    # 设置日志
    logger = setup_logging(os.path.join(output_dir, 'logs'))
    logger.info(f"开始训练 - 输出目录: {output_dir}")
    
    try:
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        num_classes = config['num_classes']
        class_names = ['background'] + config['names']
        
        logger.info(f"类别数: {num_classes}, 类别: {class_names}")
        
        # 定义数据集路径
        train_image_dir = os.path.join(data_dir, 'images', 'train')
        train_anno_file = os.path.join(data_dir, 'annotations', 'train.json')
        
        val_image_dir = os.path.join(data_dir, 'images', 'val')
        val_anno_file = os.path.join(data_dir, 'annotations', 'val.json')
        
        # 选择数据转换方式
        if use_albumentations:
            train_transform = get_transform_v2(train=True)
            val_transform = get_transform_v2(train=False)
        else:
            train_transform = get_transform(train=True)
            val_transform = get_transform(train=False)
        
        # 创建数据集
        dataset_train = InsectsDataset(train_image_dir, train_anno_file, transforms=train_transform, is_train=True)
        dataset_val = InsectsDataset(val_image_dir, val_anno_file, transforms=val_transform, is_train=False)
        
        logger.info(f"训练集大小: {len(dataset_train)}, 验证集大小: {len(dataset_val)}")
        
        # 计算类别权重
        class_weights = dataset_train.get_class_weights()
        logger.info(f"类别权重: {class_weights}")
        
        # 定义数据加载器，优化性能
        train_loader = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            collate_fn=utils.collate_fn, 
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,  # 预加载批次数
            persistent_workers=True if num_workers > 0 else False  # 保持工作线程存活
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            collate_fn=utils.collate_fn, 
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # 初始化模型
        logger.info(f"初始化模型: {backbone}")
        model = get_faster_rcnn_model(num_classes, pretrained=True, backbone=backbone)
        
        # 移动模型到GPU（如果可用）
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"使用设备: {device}")
        model.to(device)
        
        # 定义优化器
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        
        # 初始化梯度缩放器，用于混合精度训练
        scaler = GradScaler() if use_amp and device.type == 'cuda' else None
        if scaler:
            logger.info("启用混合精度训练 (AMP)")
        
        # 使用余弦退火学习率调度 - 更好的收敛效果
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/100)
        
        # TensorBoard
        tb_writer = SummaryWriter(os.path.join(output_dir, 'tb_logs'))
        
        best_map = 0.0
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logger.info(f"开始第 {epoch+1}/{num_epochs} 个周期")
            
            # 训练一个周期，传入scaler参数
            try:
                metric_logger = train_with_progress(model, optimizer, train_loader, device, epoch, num_epochs, scaler=scaler)
                
                # 存储总训练损失
                train_loss_value = metric_logger.meters['loss'].global_avg
                results['train_loss'].append(train_loss_value)
                
                # 存储损失组件
                loss_components = {}
                for name, meter in metric_logger.meters.items():
                    if name.startswith('loss_'):
                        loss_components[name] = meter.global_avg
                results['loss_components'].append(loss_components)
            except Exception as e:
                logger.error(f"训练阶段发生错误: {e}")
                logger.error(f"尝试继续进行评估...")
                train_loss_value = results['train_loss'][-1] if results['train_loss'] else 0
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            results['learning_rate'].append(current_lr)
            logger.info(f"当前学习率: {current_lr:.6f}")
            
            # 更新学习率
            lr_scheduler.step()
            
            # 在验证集上评估
            logger.info("在验证集上评估模型...")
            
            # 使用修复后的评估函数，避免原来的错误
            all_val_losses = []
            model.eval()
            
            with torch.no_grad():
                try:
                    for images, targets in tqdm(val_loader, desc="评估模型"):
                        images = list(img.to(device) for img in images)
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        # 计算损失 - 修复列表/字典处理问题
                        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=False):
                            loss_dict = model(images, targets)
                            # 处理不同形式的模型输出
                            if isinstance(loss_dict, list):
                                # 如果是列表，假设每个元素都是损失值
                                losses = sum(loss_dict)
                                # 创建一个兼容的字典以供后续处理
                                loss_dict_reduced = {"total_loss": losses}
                            else:
                                # 如果是字典，正常处理
                                loss_dict_reduced = utils.reduce_dict(loss_dict)
                                losses = sum(loss for loss in loss_dict_reduced.values())
                        
                        loss_value = losses.item() if isinstance(losses, torch.Tensor) else float(losses)
                        all_val_losses.append(loss_value)
                
                except Exception as e:
                    logger.error(f"验证损失计算发生错误: {e}")
                    if results['val_loss']:
                        all_val_losses = [results['val_loss'][-1]]
                    else:
                        all_val_losses = [0.0]
                    logger.warning("继续进行mAP评估...")
            
            # 计算验证损失
            val_loss = sum(all_val_losses) / len(all_val_losses) if all_val_losses else 0
            results['val_loss'].append(val_loss)
            
            # 计算mAP
            try:
                val_map = calculate_map(model, val_loader, device)
                results['val_map'].append(val_map)
                
                # 计算每个类别的AP
                class_ap = calculate_class_ap(model, val_loader, device, num_classes)
                results['class_ap'].append(class_ap)
                
                class_ap_str = ', '.join([f"{class_names[i+1]}: {ap:.4f}" for i, ap in class_ap.items()])
                logger.info(f"类别AP - {class_ap_str}")
            except Exception as e:
                logger.error(f"mAP计算发生错误: {e}")
                val_map = results['val_map'][-1] if results['val_map'] else 0
                results['val_map'].append(val_map)
                class_ap = results['class_ap'][-1] if results['class_ap'] else {}
                results['class_ap'].append(class_ap)
            
            # 记录到TensorBoard (使用try-except避免因TensorBoard错误中断训练)
            try:
                tb_writer.add_scalar('Loss/train', train_loss_value, epoch)
                tb_writer.add_scalar('Loss/val', val_loss, epoch)
                tb_writer.add_scalar('mAP', val_map, epoch)
                tb_writer.add_scalar('learning_rate', current_lr, epoch)
                
                # 将类别AP添加到TensorBoard
                for class_id, ap in class_ap.items():
                    if class_id+1 < len(class_names):  # 确保索引有效
                        tb_writer.add_scalar(f'AP/{class_names[class_id+1]}', ap, epoch)
                
                # 记录损失组件到TensorBoard
                for comp_name, comp_value in loss_components.items():
                    tb_writer.add_scalar(f'LossComponents/{comp_name}', comp_value, epoch)
            except Exception as e:
                logger.error(f"TensorBoard记录发生错误: {e}")
            
            # 计算本周期耗时
            epoch_time = time.time() - epoch_start_time
            
            # 打印进度
            logger.info(f"周期 {epoch+1}/{num_epochs}: 训练损失: {train_loss_value:.4f}, "
                       f"验证损失: {val_loss:.4f}, mAP: {val_map:.4f}, 用时: {epoch_time/60:.2f}分钟")
            
            # 保存最佳模型
            if val_map > best_map:
                best_map = val_map
                try:
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                    logger.info(f"保存最佳模型，mAP: {best_map:.4f}")
                except Exception as e:
                    logger.error(f"保存最佳模型时发生错误: {e}")
            
            # 保存每个周期的模型
            try:
                torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
            except Exception as e:
                logger.error(f"保存周期模型时发生错误: {e}")
            
            # 定期可视化预测结果 (使用try-except避免因可视化错误中断训练)
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                try:
                    visualize_predictions(
                        model, val_loader, device, class_names, 
                        os.path.join(output_dir, f'predictions_epoch_{epoch+1}'),
                        score_threshold=0.5, max_images=10
                    )
                except Exception as e:
                    logger.error(f"预测可视化发生错误: {e}")
                
                # 计算并绘制混淆矩阵
                try:
                    logger.info(f"正在计算周期 {epoch+1} 的混淆矩阵...")
                    cm = calculate_confusion_matrix(model, val_loader, device, num_classes, epoch+1)
                    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrices'), epoch+1)
                except Exception as e:
                    logger.error(f"混淆矩阵计算/绘制发生错误: {e}")
        
        # 训练结束后的可视化
        
        # 绘制学习率热力图
        logger.info("正在生成学习率热力图...")
        try:
            plot_learning_rate_heatmap(results['learning_rate'], output_dir)
        except Exception as e:
            logger.error(f"学习率热力图生成失败: {e}")
        
        # 绘制损失组件变化图
        if results['loss_components']:
            logger.info("正在生成损失组件变化图...")
            try:
                plot_loss_components(results, output_dir)
            except Exception as e:
                logger.error(f"损失组件变化图生成失败: {e}")
        
        # 保存最后一个模型
        try:
            torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
        except Exception as e:
            logger.error(f"保存最终模型失败: {e}")
        
        # 计算总训练时间
        total_time = time.time() - training_start_time
        logger.info(f"训练完成！总时间: {total_time/60:.2f} 分钟")
        logger.info(f"最佳mAP: {best_map:.4f}")
        
        # 保存训练结果
        try:
            with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
                # 处理不可序列化的数据
                serializable_results = {
                    'train_loss': [float(loss) for loss in results['train_loss']],
                    'val_loss': [float(loss) for loss in results['val_loss']],
                    'val_map': [float(map_val) for map_val in results['val_map']],
                    'learning_rate': [float(lr) for lr in results['learning_rate']],
                    'class_ap': [{int(k): float(v) for k, v in ap.items()} for ap in results['class_ap']],
                    'loss_components': [
                        {k: float(v) for k, v in components.items()} 
                        for components in results['loss_components']
                    ]
                }
                json.dump(serializable_results, f, indent=4)
        except Exception as e:
            logger.error(f"保存训练结果失败: {e}")
        
        # 绘制训练曲线
        try:
            save_training_plot(results, output_dir, class_names)
        except Exception as e:
            logger.error(f"绘制训练曲线失败: {e}")
        
        return model, results
    
    except Exception as e:
        logger.error(f"训练过程中发生未捕获的错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 返回尽可能多的已训练结果
        return None, results

def visualize_feature_maps(model, image_tensor, output_dir, device, layer_name=None):
    """可视化特征图（创新功能）"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # 将输入图像移至设备并添加批次维度
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 获取特定层的激活
    activations = {}
    
    # 定义钩子函数
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册钩子
    hooks = []
    
    # 不同层的激活
    if layer_name:
        # 只注册指定层
        if hasattr(model.backbone, layer_name):
            layer = getattr(model.backbone, layer_name)
            hooks.append(layer.register_forward_hook(hook_fn(layer_name)))
    else:
        # 注册骨干网络的多个层
        if hasattr(model.backbone, 'body'):
            for name, layer in model.backbone.body.named_children():
                if 'layer' in name:  # 使用ResNet的不同阶段
                    hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # 前向传播
    with torch.no_grad():
        _ = model([image_tensor])
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化激活
    for name, activation in activations.items():
        # 对于FPN来说，激活可能是特征图列表
        if isinstance(activation, list):
            for i, act in enumerate(activation):
                # 取第一个图像的特征图
                feature_map = act[0].cpu() if act.dim() > 3 else act.cpu()
                visualize_tensor(feature_map, os.path.join(output_dir, f"{name}_level{i}.png"))
        else:
            # 取批次中的第一个图像
            feature_map = activation[0].cpu()
            visualize_tensor(feature_map, os.path.join(output_dir, f"{name}.png"))
    
    return activations

def visualize_tensor(tensor, save_path, max_features=16):
    """将特征图张量可视化为图像，添加错误处理"""
    try:
        # 获取特征数量
        num_features = min(tensor.size(0), max_features)
        rows = int(num_features**0.5)
        cols = (num_features + rows - 1) // rows
        
        plt.figure(figsize=(cols * 2, rows * 2))
        
        for i in range(num_features):
            try:
                plt.subplot(rows, cols, i+1)
                # 归一化特征图以便更好地显示
                feature = tensor[i].numpy()
                vmin, vmax = feature.min(), feature.max()
                if vmax > vmin:
                    feature = (feature - vmin) / (vmax - vmin)
                plt.imshow(feature, cmap='viridis')
                plt.title(f"特征 {i+1}")
                plt.axis('off')
            except Exception as e:
                print(f"可视化特征 {i+1} 时出错: {e}")
                continue
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"特征图可视化时出错 (保存到 {save_path}): {e}")
    finally:
        # 确保图形被关闭
        try:
            plt.close()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN训练脚本')
    parser.add_argument('--data-dir', default='frcnn_code/data', help='数据集目录')
    parser.add_argument('--config', default='frcnn_code/config/insects.yaml', help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=20, help='训练周期数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'mobilenet'], help='骨干网络选择')
    parser.add_argument('--use-albumentations', action='store_true', help='使用Albumentations进行数据增强')
    parser.add_argument('--use-amp', action='store_true', help='使用混合精度训练加速')
    parser.add_argument('--visualize-features', action='store_true', help='可视化特征图')
    parser.add_argument('--workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--optimize-for-speed', action='store_true', help='优化训练速度')
    parser.add_argument('--eval-every', type=int, default=1, help='每隔多少个周期评估一次')
    
    args = parser.parse_args()
    
    # 设置CUDA参数以优化性能
    if torch.cuda.is_available() and args.optimize_for_speed:
        torch.backends.cudnn.benchmark = True
        print("启用CUDNN基准测试以提高训练速度")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 设置matplotlib字体
    # 尝试找到中文字体
    font_path = None
    for potential_path in ['SimHei.ttf', 'C:/Windows/Fonts/simhei.ttf']:
        if os.path.exists(potential_path):
            font_path = potential_path
            break
    setup_matplotlib_fonts(font_path)
    
    # 如果未指定输出目录，则创建一个以时间戳为名的目录
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('frcnn_code/results/train', f'train_{timestamp}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.output_dir, 'params.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    print("训练参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # 训练模型
    model, results = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        use_albumentations=args.use_albumentations,
        use_amp=args.use_amp,
        num_workers=args.workers
    )
    
    # 如果需要可视化特征图
    if args.visualize_features:
        print("可视化特征图...")
        # 获取一个验证集图像进行可视化
        val_image_dir = os.path.join(args.data_dir, 'images', 'val')
        val_anno_file = os.path.join(args.data_dir, 'annotations', 'val.json')
        
        # 使用非增强版本的转换
        val_transform = get_transform(train=False)
        dataset_val = InsectsDataset(val_image_dir, val_anno_file, transforms=val_transform)
        
        if len(dataset_val) > 0:
            img, _ = dataset_val[0]
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            visualize_feature_maps(model, img, os.path.join(args.output_dir, 'feature_maps'), device)
    
    print(f"训练完成！结果保存到 {args.output_dir}")
    
# 添加评估进度条的包装函数
@torch.no_grad()
def evaluate_with_progress(model, data_loader, device):
    """使用tqdm显示评估进度的包装函数"""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    
    # 使用tqdm封装数据加载器以显示进度
    pbar = tqdm(data_loader, desc="Evaluating")
    
    all_losses = []
    
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 计算损失（通过前向传播）
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()
        all_losses.append(loss_value)
        
        # 更新进度条显示
        pbar.set_postfix({'loss': f'{loss_value:.4f}'})
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    
    # 计算平均损失
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
    
    # 计算mAP（不使用COCO API，而是我们自己实现的简易版本）
    print("\n计算mAP...")
    mAP = calculate_map(model, data_loader, device)
    
    # 返回包含所需值的字典
    return {
        'loss': avg_loss,
        'mAP': mAP
    }

def calculate_map(model, data_loader, device, iou_threshold=0.5):
    """计算整体mAP（平均精度均值）"""
    model.eval()
    
    try:
        # 更可靠的获取类别数的方式
        if hasattr(data_loader.dataset, 'categories'):
            num_classes = len(data_loader.dataset.categories)
        elif hasattr(data_loader.dataset, 'coco_data') and 'categories' in data_loader.dataset.coco_data:
            num_classes = len(data_loader.dataset.coco_data['categories'])
        else:
            num_classes = 10  # 默认值
        
        # 计算每个类别的AP
        class_aps = calculate_class_ap(model, data_loader, device, num_classes + 1)  # +1 为背景类
        
        # 计算mAP（所有类别AP的均值，跳过背景类）
        mAP = np.mean([ap for class_id, ap in class_aps.items()]) if class_aps else 0
        
        return mAP
    except Exception as e:
        print(f"计算mAP时发生错误: {e}")
        # 返回一个默认值，避免中断训练
        return 0.0
def save_training_plot(results, output_dir, class_names=None):
    """保存训练和验证损失曲线，添加错误处理"""
    try:
        plt.figure(figsize=(14, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(results['train_loss'], label='训练损失')
        if 'val_loss' in results and results['val_loss']:
            plt.plot(results['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制mAP曲线
        if 'val_map' in results and results['val_map']:
            plt.subplot(2, 2, 2)
            plt.plot(results['val_map'], label='验证mAP', marker='o')
            plt.title('验证mAP')
            plt.xlabel('周期')
            plt.ylabel('mAP')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制学习率曲线
        if 'learning_rate' in results and results['learning_rate']:
            plt.subplot(2, 2, 3)
            plt.plot(results['learning_rate'], label='学习率')
            plt.title('学习率变化')
            plt.xlabel('周期')
            plt.ylabel('学习率')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
        # 绘制类别AP曲线
        if class_names and 'class_ap' in results and results['class_ap']:
            plt.subplot(2, 2, 4)
            class_ap_data = results['class_ap']
            
            for i, class_name in enumerate(class_names[1:], 0):  # 跳过背景类
                try:
                    ap_values = [epoch_data.get(i-1, 0) for epoch_data in class_ap_data if isinstance(epoch_data, dict)]
                    if ap_values:  # 确保有值可绘制
                        plt.plot(ap_values, label=f'{class_name}')
                except Exception as e:
                    print(f"绘制类别 {class_name} 的AP曲线时出错: {e}")
                    continue
                    
            plt.title('各类别AP')
            plt.xlabel('周期')
            plt.ylabel('AP')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # 将图例放在图外
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    except Exception as e:
        print(f"保存训练曲线时出错: {e}")
    finally:
        # 确保无论如何都关闭图形，避免内存泄漏
        try:
            plt.close()
        except:
            pass
        
if __name__ == "__main__":
    try:
        # 尝试导入需要的模块
        import utils
        import engine
        import transforms as T
        
        print("所有必要模块已找到，开始训练...")
        main()
    except ImportError as e:
        print(f"错误: 缺少必要模块 - {e}")
        print("请从 https://github.com/pytorch/vision/tree/main/references/detection 下载辅助文件")
        print("utils.py, engine.py, transforms.py 这三个文件应该放在同一目录下")
        sys.exit(1)