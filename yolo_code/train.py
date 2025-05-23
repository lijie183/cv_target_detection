import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import logging
from tqdm import tqdm

from config import *
from model import build_model
from data_preparation import create_data_loaders
from utils.loss import YOLOLoss
from utils.metrics import compute_ap
from utils.ema import ModelEMA
from utils.visualization import visualize_batch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname=FONT_PATH)

# 配置日志
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH / "training.log"),
        logging.StreamHandler()
    ]
)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数
        self.criterion = YOLOLoss(
            num_classes=NUM_CLASSES,
            device=device
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=EPOCHS // 5,
            T_mult=2,
            eta_min=LEARNING_RATE / 100
        )
        
        # 使用EMA模型
        if USE_EMA:
            self.ema_model = ModelEMA(model, decay=0.9998)
        
        # TensorBoard日志记录器
        self.writer = SummaryWriter(log_dir=str(LOG_PATH))
        
        # 混合精度训练
        self.scaler = amp.GradScaler()
        
        # 保存最佳模型
        self.best_map = 0.0
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'map': [],
            'learning_rates': []
        }
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        batch_losses = []
        
        # 进度条
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        
        for batch_idx, (images, targets, img_ids) in progress_bar:
            # 将图像和目标移到设备
            images = [img.to(self.device) for img in images]
            for target in targets:
                for k, v in target.items():
                    target[k] = v.to(self.device)
            
            # 梯度归零
            self.optimizer.zero_grad()
            
            # 前向传播（使用混合精度）
            with amp.autocast():
                predictions = self.model(torch.stack(images))
                loss = self.criterion(predictions, targets)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 更新EMA模型
            if USE_EMA:
                self.ema_model.update(self.model)
            
            # 更新学习率
            if batch_idx % 10 == 0:
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 累积损失
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            # 更新进度条
            progress_bar.set_description(
                f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}"
            )
            
            # 可视化每100个批次的第一个样本
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    sample_img = images[0].detach().cpu()
                    sample_target = {k: v.detach().cpu() for k, v in targets[0].items()}
                    
                    fig = visualize_batch(sample_img, sample_target, img_ids[0], epoch)
                    self.writer.add_figure(f'train/sample_batch_{batch_idx}', fig, epoch)
                    fig_path = VISUALIZATION_PATH / f"epoch_{epoch+1}_batch_{batch_idx}.png"
                    plt.savefig(str(fig_path))
                    plt.close(fig)
        
        # 一个epoch结束后，更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        
        # TensorBoard记录
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 绘制批次损失图
        self.plot_batch_losses(batch_losses, epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        if USE_EMA:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model
        
        eval_model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            
            for batch_idx, (images, targets, img_ids) in progress_bar:
                # 将图像和目标移到设备
                images = [img.to(self.device) for img in images]
                for target in targets:
                    for k, v in target.items():
                        target[k] = v.to(self.device)
                
                # 前向传播
                predictions = eval_model(torch.stack(images))
                loss = self.criterion(predictions, targets)
                
                # 累积损失
                total_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_description(
                    f"Validation | Loss: {loss.item():.6f}"
                )
                
                # 收集预测和目标
                for pred, tgt in zip(predictions, targets):
                    all_predictions.append(pred)
                    all_targets.append(tgt)
                
                # 可视化每50个批次的第一个样本
                if batch_idx % 50 == 0:
                    sample_img = images[0].detach().cpu()
                    sample_target = {k: v.detach().cpu() for k, v in targets[0].items()}
                    
                    fig = visualize_batch(sample_img, sample_target, img_ids[0], epoch, is_validation=True)
                    self.writer.add_figure(f'val/sample_batch_{batch_idx}', fig, epoch)
                    fig_path = VISUALIZATION_PATH / f"val_epoch_{epoch+1}_batch_{batch_idx}.png"
                    plt.savefig(str(fig_path))
                    plt.close(fig)
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        self.history['val_loss'].append(avg_loss)
        
        # 计算mAP
        map_score = compute_ap(all_predictions, all_targets)
        self.history['map'].append(map_score)
        
        # TensorBoard记录
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Metrics/mAP', map_score, epoch)
        
        # 保存最佳模型
        if map_score > self.best_map:
            self.best_map = map_score
            self.save_model(epoch, is_best=True)
        
        logging.info(f"Validation - Loss: {avg_loss:.6f}, mAP: {map_score:.6f}")
        
        return avg_loss, map_score
    
    def train(self, epochs):
        logging.info(f"开始训练YOLOv7-Swin，总共{epochs}个epochs")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_one_epoch(epoch)
            
            # 验证
            val_loss, map_score = self.validate(epoch)
            
            # 保存模型（每10个epoch或最后一个epoch）
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                self.save_model(epoch)
            
            # 绘制学习率曲线
            self.plot_learning_rate_curve(epoch)
            
            # 绘制损失曲线
            self.plot_loss_curves(epoch)
            
            # 绘制mAP曲线
            self.plot_map_curve(epoch)
            
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch+1}/{epochs} 完成，耗时: {datetime.timedelta(seconds=int(epoch_time))}")
            logging.info(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, mAP: {map_score:.6f}")
        
        total_time = time.time() - start_time
        logging.info(f"训练完成，总耗时: {datetime.timedelta(seconds=int(total_time))}")
        logging.info(f"最佳mAP: {self.best_map:.6f}")
        
        # 关闭TensorBoard writer
        self.writer.close()
    
    def save_model(self, epoch, is_best=False):
        """保存模型权重"""
        if is_best:
            save_path = WEIGHTS_PATH / "best_model.pth"
        else:
            save_path = WEIGHTS_PATH / f"model_epoch_{epoch+1}.pth"
        
        # 保存模型状态字典
        state_dict = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'history': self.history
        }
        
        # 如果使用EMA，保存EMA模型状态
        if USE_EMA:
            state_dict['ema_state_dict'] = self.ema_model.ema.state_dict()
        
        torch.save(state_dict, str(save_path))
        logging.info(f"模型已保存到 {save_path}")
    
    def plot_batch_losses(self, batch_losses, epoch):
        """绘制一个epoch中所有批次的损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(batch_losses, color='blue', alpha=0.8)
        plt.xlabel('批次', fontproperties=font, fontsize=12)
        plt.ylabel('损失值', fontproperties=font, fontsize=12)
        plt.title(f'第 {epoch+1} 轮训练批次损失曲线', fontproperties=font, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = VISUALIZATION_PATH / f"epoch_{epoch+1}_batch_losses.png"
        plt.savefig(str(save_path))
        
        # 添加到TensorBoard
        self.writer.add_figure('Train/batch_losses', plt.gcf(), epoch)
        
        plt.close()
    
    def plot_learning_rate_curve(self, epoch):
        """绘制学习率变化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['learning_rates'], color='green', alpha=0.8)
        plt.xlabel('优化步数', fontproperties=font, fontsize=12)
        plt.ylabel('学习率', fontproperties=font, fontsize=12)
        plt.title('学习率变化曲线', fontproperties=font, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = VISUALIZATION_PATH / f"learning_rate_curve_epoch_{epoch+1}.png"
        plt.savefig(str(save_path))
        
        # 添加到TensorBoard
        self.writer.add_figure('Train/learning_rate', plt.gcf(), epoch)
        
        plt.close()
    
    def plot_loss_curves(self, epoch):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        plt.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', alpha=0.8)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='验证损失', alpha=0.8)
        
        plt.xlabel('轮次', fontproperties=font, fontsize=12)
        plt.ylabel('损失值', fontproperties=font, fontsize=12)
        plt.title('训练和验证损失曲线', fontproperties=font, fontsize=14)
        plt.legend(prop=font)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = VISUALIZATION_PATH / f"loss_curves_epoch_{epoch+1}.png"
        plt.savefig(str(save_path))
        
        # 添加到TensorBoard
        self.writer.add_figure('Train/loss_curves', plt.gcf(), epoch)
        
        plt.close()
    
    def plot_map_curve(self, epoch):
        """绘制mAP变化曲线"""
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(self.history['map']) + 1))
        
        plt.plot(epochs, self.history['map'], 'g-', alpha=0.8)
        
        plt.xlabel('轮次', fontproperties=font, fontsize=12)
        plt.ylabel('mAP', fontproperties=font, fontsize=12)
        plt.title('平均精度(mAP)变化曲线', fontproperties=font, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = VISUALIZATION_PATH / f"map_curve_epoch_{epoch+1}.png"
        plt.savefig(str(save_path))
        
        # 添加到TensorBoard
        self.writer.add_figure('Train/map_curve', plt.gcf(), epoch)
        
        plt.close()

def main():
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # 构建模型
    model = build_model()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE
    )
    
    # 开始训练
    trainer.train(EPOCHS)

if __name__ == "__main__":
    main()