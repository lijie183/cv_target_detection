import os
import yaml
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
import argparse
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import utils
    import transforms as T
    from engine import evaluate
except ImportError:
    print("缺少必要模块。请从PyTorch vision/references/detection示例中获取必要的工具文件。")
    sys.exit(1)

class InsectsDataset(torch.utils.data.Dataset):
    def __init__(self, root, coco_annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        
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
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
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
            'file_name': img_info['file_name']  # 添加文件名以便以后参考
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.images)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_faster_rcnn_model(num_classes):
    # 加载预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # 替换分类器头以匹配我们的类别数量
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def evaluate_model(model, data_loader, device, output_dir):
    # 在验证集上评估模型
    print("开始模型评估...")
    coco_evaluator = evaluate(model, data_loader, device=device)
    
    # 打印评估结果
    print("\n评估结果:")
    stats = coco_evaluator.coco_eval['bbox'].stats
    
    metrics = {
        'AP@[0.5:0.95]': stats[0],
        'AP@0.5': stats[1],
        'AP@0.75': stats[2],
        'AP_small': stats[3],
        'AP_medium': stats[4],
        'AP_large': stats[5],
        'AR@[0.5:0.95]': stats[6],
        'AR@0.5': stats[7],
        'AR@0.75': stats[8],
        'AR_small': stats[9],
        'AR_medium': stats[10],
        'AR_large': stats[11]
    }
    
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # 保存评估结果
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    # 绘制主要指标
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(metrics)), list(metrics.values()), align='center')
    plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right')
    plt.title('COCO Evaluation Metrics')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'))
    plt.close()
    
    return metrics, coco_evaluator

def visualize_results(model, data_loader, device, class_names, output_dir, num_images=20, conf_threshold=0.5):
    print(f"可视化 {num_images} 张图像的预测结果...")
    model.eval()
    
    # 创建输出目录
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 评估并可视化
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader)):
            if i >= num_images:
                break
            
            # 重新排列目标框和标签以便于使用，每个目标一个
            boxes_all = []
            labels_all = []
            scores_all = []
            images_all = []
            
            # 移动图像到设备
            images = list(img.to(device) for img in images)
            
            # 进行预测
            outputs = model(images)
            
            # 处理每个图像
            for j, (img, target, output) in enumerate(zip(images, targets, outputs)):
                # 转回CPU并转换为numpy图像
                img_np = img.cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # 创建可视化图像
                vis_img = img_np.copy()
                
                # 绘制真实边界框
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                for box, label_idx in zip(gt_boxes, gt_labels):
                    box = box.astype(np.int32)
                    label = class_names[label_idx]
                    cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(vis_img, label, (box[0], box[1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 保存带真实标签的图像
                gt_filename = os.path.join(vis_dir, f"img_{i}_{j}_gt.jpg")
                cv2.imwrite(gt_filename, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
                # 绘制预测边界框
                pred_vis_img = img_np.copy()
                
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                
                # 使用阈值过滤预测结果
                mask = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]
                
                for box, label_idx, score in zip(pred_boxes, pred_labels, pred_scores):
                    box = box.astype(np.int32)
                    label = class_names[label_idx]
                    cv2.rectangle(pred_vis_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(pred_vis_img, f"{label} {score:.2f}", (box[0], box[1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 保存带预测标签的图像
                pred_filename = os.path.join(vis_dir, f"img_{i}_{j}_pred.jpg")
                cv2.imwrite(pred_filename, cv2.cvtColor(pred_vis_img, cv2.COLOR_RGB2BGR))
                
                # 同时显示真实和预测结果
                compare_img = np.hstack((vis_img, pred_vis_img))
                compare_filename = os.path.join(vis_dir, f"img_{i}_{j}_compare.jpg")
                cv2.imwrite(compare_filename, cv2.cvtColor(compare_img, cv2.COLOR_RGB2BGR))
    
    print(f"可视化结果已保存到: {vis_dir}")

def analyze_errors(model, data_loader, device, class_names, output_dir, iou_threshold=0.5, conf_threshold=0.5):
    print("分析检测错误...")
    model.eval()
    
    # 创建输出目录
    error_dir = os.path.join(output_dir, 'error_analysis')
    os.makedirs(error_dir, exist_ok=True)
    
    # 收集错误统计信息
    class_tp = {cls: 0 for cls in class_names[1:]}  # 不包括背景
    class_fp = {cls: 0 for cls in class_names[1:]}
    class_fn = {cls: 0 for cls in class_names[1:]}
    
    # 完全错检的图像
    false_positive_images = []
    # 完全漏检的图像
    false_negative_images = []
    # 混淆矩阵
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int32)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader)):
            # 移动图像到设备
            images = list(img.to(device) for img in images)
            
            # 进行预测
            outputs = model(images)
            
            # 处理每个图像
            for j, (img, target, output) in enumerate(zip(images, targets, outputs)):
                # 获取真实边界框和类别
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                gt_classes = [class_names[l] for l in gt_labels]
                
                # 获取预测边界框和类别
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                
                # 使用阈值过滤预测结果
                mask = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]
                pred_classes = [class_names[l] for l in pred_labels]
                
                # 跟踪匹配状态
                gt_matched = [False] * len(gt_boxes)
                pred_matched = [False] * len(pred_boxes)
                
                # 计算IoU矩阵
                ious = np.zeros((len(gt_boxes), len(pred_boxes)))
                for gt_idx, gt_box in enumerate(gt_boxes):
                    for pred_idx, pred_box in enumerate(pred_boxes):
                        ious[gt_idx, pred_idx] = compute_iou(gt_box, pred_box)
                
                # 匹配真实框和预测框
                for gt_idx, gt_label in enumerate(gt_labels):
                    gt_class = class_names[gt_label]
                    
                    # 查找匹配的预测框
                    best_iou = iou_threshold
                    best_pred_idx = -1
                    
                    for pred_idx, pred_label in enumerate(pred_labels):
                        if pred_matched[pred_idx]:
                            continue
                        
                        # 检查是否是相同类别并且IoU足够高
                        if pred_label == gt_label and ious[gt_idx, pred_idx] > best_iou:
                            best_iou = ious[gt_idx, pred_idx]
                            best_pred_idx = pred_idx
                    
                    # 如果找到匹配
                    if best_pred_idx >= 0:
                        gt_matched[gt_idx] = True
                        pred_matched[best_pred_idx] = True
                        class_tp[gt_class] += 1
                        
                        # 在混淆矩阵中标记真阳性
                        confusion_matrix[gt_label, pred_labels[best_pred_idx]] += 1
                    else:
                        # 假阴性(漏检)
                        class_fn[gt_class] += 1
                        
                        # 在混淆矩阵中标记假阴性
                        confusion_matrix[gt_label, 0] += 1  # 0表示背景，即未检测
                
                # 处理假阳性(误检)
                for pred_idx, pred_label in enumerate(pred_labels):
                    if not pred_matched[pred_idx]:
                        pred_class = class_names[pred_label]
                        class_fp[pred_class] += 1
                        
                        # 在混淆矩阵中标记假阳性
                        confusion_matrix[0, pred_label] += 1  # 0表示背景，误检为某类
                
                # 检查是否有完全错检或漏检的情况
                file_name = target.get('file_name', f"img_{i}_{j}")
                
                if len(gt_boxes) > 0 and len(pred_boxes) == 0:
                    false_negative_images.append((file_name, len(gt_boxes)))
                
                if len(gt_boxes) == 0 and len(pred_boxes) > 0:
                    false_positive_images.append((file_name, len(pred_boxes)))
    
    # 计算每个类别的精确率、召回率和F1分数
    class_metrics = {}
    for cls in class_names[1:]:  # 跳过背景类
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # 保存错误分析结果
    results = {
        'class_metrics': class_metrics,
        'false_positive_images': false_positive_images,
        'false_negative_images': false_negative_images
    }
    
    with open(os.path.join(error_dir, 'error_analysis.json'), 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    # 绘制每个类别的精确率、召回率和F1分数
    classes = list(class_metrics.keys())
    precisions = [class_metrics[cls]['precision'] for cls in classes]
    recalls = [class_metrics[cls]['recall'] for cls in classes]
    f1s = [class_metrics[cls]['f1'] for cls in classes]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Class-wise Precision, Recall, and F1 Score')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, 'class_metrics.png'))
    plt.close()
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 设置刻度标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # 在矩阵中显示数值
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, 'confusion_matrix.png'))
    plt.close()
    
    return results

def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 如果没有交集
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN模型评估')
    parser.add_argument('--model', required=True, type=str, help='模型权重文件路径')
    parser.add_argument('--data-dir', default='frcnn_code/data', type=str, help='数据目录')
    parser.add_argument('--config', default='frcnn_code/config/insects.yaml', type=str, help='配置文件')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作线程数')
    parser.add_argument('--visualize', type=int, default=10, help='可视化的图像数量')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 如果未指定输出目录，则创建一个以时间戳为名的目录
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('frcnn_code/results/evaluation', f'eval_{timestamp}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.output_dir, 'eval_params.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = config['num_classes'] + 1  # 加1是因为包括背景类
    class_names = ['background'] + config['names']
    
    print(f"类别数: {num_classes}, 类别: {class_names}")
    
    # 加载模型
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"从 {args.model} 加载模型")
    
    # 创建验证数据集和数据加载器
    val_image_dir = os.path.join(args.data_dir, 'images', 'val')
    val_anno_file = os.path.join(args.data_dir, 'annotations', 'val.json')
    
    dataset_val = InsectsDataset(val_image_dir, val_anno_file, transforms=get_transform(train=False))
    
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=utils.collate_fn)
    
    print(f"验证集大小: {len(dataset_val)}")
    
    # 评估模型
    metrics, coco_evaluator = evaluate_model(model, val_loader, device, args.output_dir)
    
    # 可视化一些预测结果
    visualize_results(model, val_loader, device, class_names, args.output_dir, 
                     num_images=args.visualize, conf_threshold=args.conf_threshold)
    
    # 分析错误
    error_analysis = analyze_errors(model, val_loader, device, class_names, 
                                  args.output_dir, conf_threshold=args.conf_threshold)
    
    print(f"评估完成！结果保存到 {args.output_dir}")

if __name__ == "__main__":
    main()