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
        # 将numpy值转换为原生Python类型以便JSON序列化
        serializable_metrics = {
            'mAP': float(metrics['mAP']),
            'per_class_ap': {cls: float(ap) for cls, ap in metrics['per_class_ap'].items()}
        }
        json.dump(serializable_metrics, f, indent=4)
    
    # 绘制并保存可视化图表
    
    # 1. 各类别AP柱状图
    plt.figure(figsize=(14, 10))
    classes = list(metrics['per_class_ap'].keys())
    ap_values = list(metrics['per_class_ap'].values())
    
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
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics', 'per_class_ap.png'))
    plt.close()
    
    # 2. 绘制精确率-召回率曲线
    plt.figure(figsize=(12, 8))
    
    # 选择最重要的几个类别或AP值最高的几个类别
    num_classes_to_plot = min(5, len(classes))
    selected_classes = [classes[i] for i in sorted_indices[:num_classes_to_plot]]
    
    for cls in selected_classes:
        precision = metrics['per_class_pr'][cls]['precision']
        recall = metrics['per_class_pr'][cls]['recall']
        plt.plot(recall, precision, label=f'{cls} (AP: {metrics["per_class_ap"][cls]:.2f})', linewidth=2)
    
    plt.title('精确率-召回率曲线', fontsize=16)
    plt.xlabel('召回率 (Recall)', fontsize=14)
    plt.ylabel('精确率 (Precision)', fontsize=14)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics', 'precision_recall_curves.png'))
    plt.close()
    
    # 3. 混淆矩阵
    plt.figure(figsize=(14, 12))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar(label='样本数量')
    plt.title('混淆矩阵', fontsize=16)
    plt.xlabel('预测类别', fontsize=14)
    plt.ylabel('真实类别', fontsize=14)
    
    # 在坐标轴上添加类别名称
    all_classes = ['background'] + class_names
    plt.xticks(range(len(all_classes)), all_classes, rotation=90)
    plt.yticks(range(len(all_classes)), all_classes)
    
    # 在单元格中添加数值
    for i in range(len(all_classes)):
        for j in range(len(all_classes)):
            if confusion_matrix[i, j] > 0:
                plt.text(j, i, str(confusion_matrix[i, j]),
                        ha="center", va="center", 
                        color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics', 'confusion_matrix.png'))
    plt.close()
    
    return metrics

def test_model(data_dir, model_path, output_dir='mamba_code/results/test', device='cuda'):
    """主测试函数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据
    _, _, test_loader = get_data_loaders(data_dir, batch_size=4)
    
    # 创建并加载模型
    model = create_model(num_classes=len(VOC_CLASSES) + 1)  # +1表示背景类
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    print(f"模型已从 {model_path} 加载")
    print(f"在 {len(test_loader.dataset)} 张图像上进行测试")
    
    # 在测试集上评估模型
    metrics = evaluate_model(model, test_loader, device, output_dir)
    
    print(f"测试mAP: {metrics['mAP']:.4f}")
    
    # 打印每个类别的结果
    print("\n各类别平均精度 (AP):")
    for cls, ap in sorted(metrics['per_class_ap'].items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {ap:.4f}")
    
    return metrics

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置数据目录和模型路径
    data_dir = "../data"
    model_path = "mamba_code/results/models/model_final.pth"
    output_dir = "mamba_code/results/test"
    
    # 测试模型
    metrics = test_model(
        data_dir=data_dir,
        model_path=model_path,
        output_dir=output_dir,
        device=device
    )
    
    print("测试成功完成!")