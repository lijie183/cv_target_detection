import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import logging
from tqdm import tqdm
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import cv2

from config import *
from model import build_model
from data_preparation import create_data_loaders
from utils.metrics import compute_ap, box_iou, ap_per_class, xywh2xyxy
from utils.visualization import visualize_predictions

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname=FONT_PATH)

# 配置日志
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH / "validation.log"),
        logging.StreamHandler()
    ]
)

class Validator:
    def __init__(self, model, val_loader, device, confidence_threshold=0.25, iou_threshold=0.45):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 确保结果目录存在
        self.validation_results_path = METRIC_PATH / "validation"
        self.validation_images_path = VISUALIZATION_PATH / "validation"
        self.validation_results_path.mkdir(parents=True, exist_ok=True)
        self.validation_images_path.mkdir(parents=True, exist_ok=True)
        
        # 存储结果
        self.class_ap = {cls_name: 0.0 for cls_name in VOC_CLASSES}
        self.predictions = []
        self.targets = []
        self.confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
    def validate(self):
        logging.info("开始验证...")
        self.model.eval()
        
        # 收集所有类别的预测和真实标签
        all_pred_boxes = []  # 所有预测框
        all_pred_labels = []  # 所有预测标签
        all_pred_scores = []  # 所有预测分数
        all_true_boxes = []  # 所有真实框
        all_true_labels = []  # 所有真实标签
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            
            for batch_idx, (images, targets, img_ids) in progress_bar:
                # 将图像和目标移到设备
                images = [img.to(self.device) for img in images]
                for target in targets:
                    for k, v in target.items():
                        target[k] = v.to(self.device)
                
                # 前向传播获取预测结果
                predictions = self.model(torch.stack(images))
                
                # 处理每张图像的预测和目标
                for i, (pred, target, img_id) in enumerate(zip(predictions, targets, img_ids)):
                    # 筛选满足置信度阈值的检测结果
                    mask = pred[:, 4] > self.confidence_threshold
                    pred_filtered = pred[mask]
                    
                    if len(pred_filtered) > 0:
                        # 获取预测框和类别
                        pred_boxes = pred_filtered[:, :4]  # x, y, w, h 格式
                        pred_scores = pred_filtered[:, 4]
                        pred_class_scores = pred_filtered[:, 5:]
                        pred_labels = torch.argmax(pred_class_scores, dim=1)
                        
                        # 转换为 xyxy 格式
                        pred_boxes_xyxy = xywh2xyxy(pred_boxes)
                        
                        # 保存预测结果
                        for box, label, score in zip(pred_boxes_xyxy.cpu().numpy(), 
                                                    pred_labels.cpu().numpy(), 
                                                    pred_scores.cpu().numpy()):
                            all_pred_boxes.append(box)
                            all_pred_labels.append(label)
                            all_pred_scores.append(score)
                    
                    # 获取真实标注
                    true_boxes = target['boxes']  # yolo 格式 (x_center, y_center, width, height)
                    true_labels = target['labels']
                    
                    # 转换为 xyxy 格式
                    true_boxes_xyxy = xywh2xyxy(true_boxes)
                    
                    for box, label in zip(true_boxes_xyxy.cpu().numpy(), true_labels.cpu().numpy()):
                        all_true_boxes.append(box)
                        all_true_labels.append(label)
                    
                    # 可视化每50批次的第一个样本
                    if batch_idx % 50 == 0 and i == 0:
                        sample_img = images[i].detach().cpu()
                        sample_pred = pred_filtered.detach().cpu() if len(pred_filtered) > 0 else None
                        sample_target = {k: v.detach().cpu() for k, v in target.items()}
                        
                        # 可视化预测结果
                        fig = visualize_predictions(
                            sample_img, 
                            sample_pred, 
                            sample_target, 
                            img_id,
                            VOC_CLASSES
                        )
                        
                        save_path = self.validation_images_path / f"val_sample_{batch_idx}.png"
                        plt.savefig(str(save_path))
                        plt.close(fig)
        
        # 转换为numpy数组
        all_pred_boxes = np.array(all_pred_boxes)
        all_pred_labels = np.array(all_pred_labels)
        all_pred_scores = np.array(all_pred_scores)
        all_true_boxes = np.array(all_true_boxes)
        all_true_labels = np.array(all_true_labels)
        
        # 计算各类别AP和mAP
        stats = []
        
        # 检查是否有预测结果
        if len(all_pred_boxes) > 0:
            # 计算各类别的AP
            ap, p, r = ap_per_class(
                tp=all_pred_boxes,
                conf=all_pred_scores,
                pred_cls=all_pred_labels,
                target_cls=all_true_labels,
                target_boxes=all_true_boxes,
                iou_thres=self.iou_threshold
            )
            
            # 存储各类别AP
            for i, class_name in enumerate(VOC_CLASSES):
                self.class_ap[class_name] = ap[i] if i < len(ap) else 0.0
            
            # 计算mAP
            map_score = ap.mean()
            
            # 打印结果
            logging.info(f"验证完成，mAP@{self.iou_threshold}: {map_score:.4f}")
            
            # 可视化各类别的AP
            self.visualize_class_ap()
            
            # 可视化PR曲线
            self.visualize_pr_curves(p, r)
            
            # 可视化混淆矩阵
            self.calculate_confusion_matrix(all_pred_labels, all_true_labels)
            self.visualize_confusion_matrix()
            
            # 保存验证结果
            self.save_validation_results(map_score)
        else:
            logging.warning("没有检测到任何目标，无法计算AP")
            map_score = 0.0
        
        return map_score
    
    def visualize_class_ap(self):
        """可视化各类别的AP"""
        plt.figure(figsize=(14, 8))
        classes = list(self.class_ap.keys())
        ap_values = list(self.class_ap.values())
        
        # 创建条形图
        bars = plt.bar(classes, ap_values, color='skyblue')
        
        # 在条形上方显示具体数值
        for bar, ap in zip(bars, ap_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{ap:.3f}', ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.xlabel('类别', fontproperties=font, fontsize=12)
        plt.ylabel('AP', fontproperties=font, fontsize=12)
        plt.title('各类别的平均精度(AP)', fontproperties=font, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.validation_results_path / "class_ap.png"
        plt.savefig(str(save_path))
        plt.close()
    
    def visualize_pr_curves(self, p, r):
        """可视化PR曲线"""
        plt.figure(figsize=(12, 10))
        
        # 绘制每个类别的PR曲线
        for i, class_name in enumerate(VOC_CLASSES):
            if i < len(p) and i < len(r):
                plt.plot(r[i], p[i], linewidth=2, label=f"{class_name} (AP={self.class_ap[class_name]:.3f})")
        
        plt.xlabel('召回率', fontproperties=font, fontsize=12)
        plt.ylabel('精确率', fontproperties=font, fontsize=12)
        plt.title('各类别的精确率-召回率曲线', fontproperties=font, fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(linestyle='--', alpha=0.6)
        plt.legend(loc='lower left', prop=font)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.validation_results_path / "pr_curves.png"
        plt.savefig(str(save_path))
        plt.close()
        
        # 单独绘制每个类别的PR曲线，便于观察
        os.makedirs(str(self.validation_results_path / "pr_curves"), exist_ok=True)
        
        for i, class_name in enumerate(VOC_CLASSES):
            if i < len(p) and i < len(r):
                plt.figure(figsize=(8, 6))
                plt.plot(r[i], p[i], linewidth=2, color='blue')
                plt.xlabel('召回率', fontproperties=font, fontsize=12)
                plt.ylabel('精确率', fontproperties=font, fontsize=12)
                plt.title(f'{class_name} 精确率-召回率曲线 (AP={self.class_ap[class_name]:.3f})', fontproperties=font, fontsize=14)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.grid(linestyle='--', alpha=0.6)
                plt.tight_layout()
                
                # 保存图像
                save_path = self.validation_results_path / "pr_curves" / f"{class_name}_pr_curve.png"
                plt.savefig(str(save_path))
                plt.close()
    
    def calculate_confusion_matrix(self, pred_labels, true_labels):
        """计算混淆矩阵"""
        self.confusion_matrix = confusion_matrix(
            true_labels, 
            pred_labels, 
            labels=list(range(NUM_CLASSES))
        )
    
    def visualize_confusion_matrix(self):
        """可视化混淆矩阵"""
        plt.figure(figsize=(16, 14))
        
        # 计算归一化的混淆矩阵
        cm_normalized = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # 使用seaborn绘制热图
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=VOC_CLASSES,
            yticklabels=VOC_CLASSES
        )
        
        plt.xlabel('预测类别', fontproperties=font, fontsize=12)
        plt.ylabel('真实类别', fontproperties=font, fontsize=12)
        plt.title('混淆矩阵', fontproperties=font, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.validation_results_path / "confusion_matrix.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_validation_results(self, map_score):
        """保存验证结果到JSON文件"""
        results = {
            'mAP': float(map_score),
            'class_AP': {k: float(v) for k, v in self.class_ap.items()},
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold
        }
        
        with open(str(self.validation_results_path / "validation_results.json"), 'w') as f:
            json.dump(results, f, indent=4)

def main():
    # 创建数据加载器
    _, val_loader, _ = create_data_loaders()
    
    # 构建模型
    model = build_model()
    
    # 加载最佳模型权重
    best_model_path = WEIGHTS_PATH / "best_model.pth"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        
        # 如果有EMA模型状态，使用EMA模型
        if 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'])
            logging.info("已加载EMA模型权重")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("已加载模型权重")
    else:
        logging.warning(f"找不到模型权重文件: {best_model_path}，使用随机初始化的模型进行验证")
    
    # 创建验证器
    validator = Validator(
        model=model,
        val_loader=val_loader,
        device=DEVICE,
        confidence_threshold=CONF_THRESH,
        iou_threshold=IOU_THRESH
    )
    
    # 执行验证
    map_score = validator.validate()
    
    logging.info(f"验证完成，mAP@{IOU_THRESH}: {map_score:.4f}")

if __name__ == "__main__":
    main()