import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import logging
from tqdm import tqdm
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import time

from config import *
from model import build_model
from data_preparation import create_data_loaders
from utils.metrics import xywh2xyxy, non_max_suppression
from utils.visualization import visualize_test_predictions

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname=FONT_PATH)

# 配置日志
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH / "test.log"),
        logging.StreamHandler()
    ]
)

class Tester:
    def __init__(self, model, test_loader, device, confidence_threshold=0.25, iou_threshold=0.45):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 确保结果目录存在
        self.test_results_path = PREDICTION_PATH
        self.test_results_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.detection_images_path = self.test_results_path / "detection_images"
        self.detection_images_path.mkdir(parents=True, exist_ok=True)
        
        # 存储性能指标
        self.metrics = {
            'average_precision': 0.0,
            'inference_time': 0.0,
            'fps': 0.0
        }
    
    def test(self):
        logging.info("开始测试...")
        self.model.eval()
        
        # 存储检测结果
        all_detections = []
        inference_times = []
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            
            for batch_idx, (images, img_ids, orig_sizes) in progress_bar:
                # 将图像移到设备
                images = images.to(self.device)
                
                # 计时开始
                start_time = time.time()
                
                # 前向传播获取预测结果
                predictions = self.model(images)
                
                # 应用NMS
                predictions = non_max_suppression(
                    predictions, 
                    conf_thres=self.confidence_threshold,
                    iou_thres=self.iou_threshold
                )
                
                # 计时结束
                end_time = time.time()
                inference_time = (end_time - start_time) / images.size(0)  # 平均每张图像的推理时间
                inference_times.append(inference_time)
                
                # 处理每张图像的预测结果
                for i, (pred, img_id, orig_size) in enumerate(zip(predictions, img_ids, orig_sizes)):
                    # 创建检测结果字典
                    detection = {
                        'image_id': img_id,
                        'detections': []
                    }
                    
                    if pred is not None and len(pred) > 0:
                        # 调整检测框到原始图像尺寸
                        pred[:, :4] = scale_boxes(images.shape[2:], pred[:, :4], orig_size)
                        
                        # 保存每个检测结果
                        for *xyxy, conf, cls_id in pred:
                            box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            category_id = int(cls_id.item())
                            score = float(conf.item())
                            
                            detection['detections'].append({
                                'category_id': category_id,
                                'category_name': VOC_CLASSES[category_id],
                                'bbox': box,
                                'score': score
                            })
                    
                    # 添加到所有检测结果
                    all_detections.append(detection)
                    
                    # 可视化前100个测试图像
                    if batch_idx * images.size(0) + i < 100:
                        # 加载原始图像
                        img_path = TEST_DATA_PATH / "JPEGImages" / f"{img_id}.jpg"
                        if Path(img_path).exists():
                            original_img = cv2.imread(str(img_path))
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                            
                            # 可视化检测结果
                            result_img = visualize_test_predictions(
                                original_img.copy(), 
                                detection['detections']
                            )
                            
                            # 保存结果图像
                            save_path = self.detection_images_path / f"{img_id}_detection.jpg"
                            cv2.imwrite(str(save_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        # 计算平均推理时间和FPS
        avg_inference_time = np.mean(inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        self.metrics['inference_time'] = avg_inference_time
        self.metrics['fps'] = avg_fps
        
        logging.info(f"测试完成")
        logging.info(f"平均推理时间: {avg_inference_time*1000:.2f} ms")
        logging.info(f"平均FPS: {avg_fps:.2f}")
        
        # 保存检测结果和性能指标
        self.save_test_results(all_detections)
        self.visualize_performance_metrics()
        
        return all_detections
    
    def save_test_results(self, detections):
        """保存测试结果到JSON文件"""
        # 保存所有检测结果
        with open(str(self.test_results_path / "detections.json"), 'w') as f:
            json.dump(detections, f, indent=4)
        
        # 保存性能指标
        with open(str(self.test_results_path / "performance_metrics.json"), 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def visualize_performance_metrics(self):
        """可视化性能指标"""
        # 绘制推理时间柱状图
        plt.figure(figsize=(10, 6))
        metrics = ['推理时间(ms)', 'FPS']
        values = [self.metrics['inference_time'] * 1000, self.metrics['fps']]
        colors = ['blue', 'green']
        
        bars = plt.bar(metrics, values, color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.ylabel('数值', fontproperties=font, fontsize=12)
        plt.title('模型性能指标', fontproperties=font, fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.test_results_path / "performance_metrics.png"
        plt.savefig(str(save_path))
        plt.close()

def scale_boxes(img_size, boxes, target_size):
    """将预测框从模型输入尺寸缩放到原始图像尺寸"""
    # img_size: 模型输入尺寸 (height, width)
    # boxes: 预测框 (xyxy格式)
    # target_size: 原始图像尺寸 (height, width)
    
    ratio_w = target_size[1] / img_size[1]
    ratio_h = target_size[0] / img_size[0]
    
    scaled_boxes = boxes.clone()
    scaled_boxes[:, 0] *= ratio_w
    scaled_boxes[:, 2] *= ratio_w
    scaled_boxes[:, 1] *= ratio_h
    scaled_boxes[:, 3] *= ratio_h
    
    return scaled_boxes

def main():
    # 创建数据加载器
    _, _, test_loader = create_data_loaders()
    
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
        logging.warning(f"找不到模型权重文件: {best_model_path}，使用随机初始化的模型进行测试")
    
    # 创建测试器
    tester = Tester(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        confidence_threshold=CONF_THRESH,
        iou_threshold=IOU_THRESH
    )
    
    # 执行测试
    detections = tester.test()
    
    logging.info(f"测试完成，检测到 {sum(len(d['detections']) for d in detections)} 个目标")

if __name__ == "__main__":
    main()