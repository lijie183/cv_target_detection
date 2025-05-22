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