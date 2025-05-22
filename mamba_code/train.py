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