import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from ultralytics.models import YOLO
import cv2
import argparse
import logging
import matplotlib
import platform
import traceback
from datetime import datetime

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
        return True
    except Exception as e:
        print(f"字体设置失败: {e}, 可能会导致中文显示异常")
        return False

def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def predict_image(model, img_path, class_names, output_dir, conf_threshold=0.25):
    """预测单张图像"""
    try:
        img_path = Path(img_path)
        img_name = img_path.name
        base_name = img_path.stem
        
        # 加载图像
        img = Image.open(img_path)
        
        # 进行预测
        results = model(img_path, conf=conf_threshold)
        
        # 获取颜色 - 不再转换为0-255范围，让matplotlib自己处理
        colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
        # 保持颜色在0-1范围内，适合matplotlib
        
        # 可视化结果
        result_img = results[0].plot(labels=True, line_width=2)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # 保存结果
        output_path = Path(output_dir) / f"pred_{base_name}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        # 生成检测结果统计图
        plt.figure(figsize=(12, 6))
        
        # 绘制原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(img))
        plt.title('原始图像')
        plt.axis('off')
        
        # 绘制检测结果
        plt.subplot(1, 2, 2)
        plt.imshow(result_img)
        plt.title('检测结果')
        plt.axis('off')
        
        plt.tight_layout()
        vis_output_path = Path(output_dir) / f"vis_{base_name}.png"
        plt.savefig(vis_output_path, dpi=300)
        plt.close()
        
        # 统计检测到的目标
        boxes = results[0].boxes
        class_counts = {}
        
        for box in boxes:
            cls_id = int(box.cls[0])
            if 0 <= cls_id < len(class_names):  # 确保类别ID有效
                class_name = class_names[cls_id]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        
        # 绘制目标统计图
        if class_counts:
            plt.figure(figsize=(10, 6))
            plt.bar(class_counts.keys(), class_counts.values())
            plt.title(f'图像中检测到的目标数量统计')
            plt.xlabel('目标类别')
            plt.ylabel('数量')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            stats_output_path = Path(output_dir) / f"stats_{base_name}.png"
            plt.savefig(stats_output_path, dpi=300)
            plt.close()
        
        return results[0], class_counts
    
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {e}")
        traceback.print_exc()
        return None, {}

def predict_folder(model, img_folder, class_names, output_dir, conf_threshold=0.25):
    """预测文件夹中的所有图像"""
    img_folder = Path(img_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 获取所有图像文件
    img_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        img_files.extend(list(img_folder.glob(f'*{ext}')))
        img_files.extend(list(img_folder.glob(f'*{ext.upper()}')))
    
    if not img_files:
        print(f"警告: 在 {img_folder} 中未找到图像文件")
        return {}
    
    all_class_counts = {}
    total_objects = 0
    processed_count = 0
    
    print(f"开始预测 {len(img_files)} 张图像...")
    for img_path in img_files:
        result, class_counts = predict_image(model, img_path, class_names, output_dir, conf_threshold)
        
        # 累计统计
        for cls, count in class_counts.items():
            if cls not in all_class_counts:
                all_class_counts[cls] = 0
            all_class_counts[cls] += count
            total_objects += count
        
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == len(img_files):
            print(f"已处理 {processed_count}/{len(img_files)} 张图像")
    
    # 绘制总体统计图表
    if all_class_counts:
        try:
            # 柱状图
            plt.figure(figsize=(12, 6))
            plt.bar(all_class_counts.keys(), all_class_counts.values())
            plt.title(f'测试集中检测到的目标类别分布 (总计: {total_objects})')
            plt.xlabel('目标类别')
            plt.ylabel('数量')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(str(output_dir / "class_distribution_bar.png"), dpi=300)
            plt.close()
            
            # 饼图
            plt.figure(figsize=(10, 10))
            plt.pie(
                list(all_class_counts.values()),
                labels=list(all_class_counts.keys()),
                autopct='%1.1f%%',
                startangle=90
            )
            plt.title('测试集中各类别占比')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(str(output_dir / "class_distribution_pie.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"生成统计图表时出错: {e}")
            traceback.print_exc()
    
    return all_class_counts

def find_latest_model(base_dir):
    """查找所有可能的模型路径，返回最新的一个"""
    # 模型可能的位置
    base_dir = Path(base_dir)
    possible_locations = [
        # 模型目录中的模型
        list(base_dir.glob('**/models/*.pt')),
        # train目录中的weights目录
        list(base_dir.glob('**/train*/weights/*.pt')),
    ]
    
    all_models = []
    for location in possible_locations:
        all_models.extend(location)
    
    if not all_models:
        return None
    
    # 按修改时间排序，返回最新的
    return sorted(all_models, key=os.path.getmtime)[-1]

def main():
    parser = argparse.ArgumentParser(description="使用训练好的YOLOv8模型进行预测")
    parser.add_argument('--img_folder', type=str, help='要预测的图像文件夹路径')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    args = parser.parse_args()
    
    # 使用Path处理路径
    base_dir = Path(__file__).resolve().parent.parent.parent
    yolo_dir = base_dir / "yolo_code"
    
    # 设置目录
    config_path = yolo_dir / "config" / "insects.yaml"
    results_dir = yolo_dir / "results"
    models_dir = results_dir / "models"
    predictions_dir = results_dir / "predictions"
    logs_dir = results_dir / "logs"
    
    # 确保目录存在
    for directory in [predictions_dir, logs_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # 尝试加载本地字体或项目根目录下的字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 设置日志
    logger = setup_logging(logs_dir)
    logger.info(f"系统环境: {platform.system()} {platform.release()}")
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        class_names = config['names']
        logger.info(f"加载类别信息: {class_names}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return
    
    # 加载模型
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.warning(f"指定的模型文件不存在: {model_path}, 将尝试查找其他模型")
            model_path = None
    
    if not model_path:
        # 尝试加载models目录中的best_model.pt
        default_model_path = models_dir / "best_model.pt"
        if default_model_path.exists():
            model_path = default_model_path
            logger.info(f"使用默认模型: {model_path}")
        else:
            # 尝试查找最新的模型
            model_path = find_latest_model(yolo_dir)
            if model_path:
                logger.info(f"找到最新模型: {model_path}")
            else:
                logger.error("未找到可用的模型文件")
                return
    
    logger.info(f"加载模型: {model_path}")
    try:
        model = YOLO(model_path)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return
    
    # 确定预测图像文件夹
    img_folder = args.img_folder
    if not img_folder:
        # 默认使用验证集进行预测
        img_folder = yolo_dir / "data" / "images" / "val"
        logger.info(f"未指定图像文件夹，使用默认验证集: {img_folder}")
    else:
        img_folder = Path(img_folder)
    
    if not img_folder.exists():
        logger.error(f"图像文件夹不存在: {img_folder}")
        return
    
    logger.info(f"开始预测文件夹中的图像: {img_folder}")
    
    try:
        all_class_counts = predict_folder(model, img_folder, class_names, predictions_dir, args.conf)
        
        # 保存统计结果
        stats_file = predictions_dir / "prediction_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("类别\t数量\n")
            total = 0
            for cls, count in all_class_counts.items():
                f.write(f"{cls}\t{count}\n")
                total += count
            f.write(f"总计\t{total}\n")
        
        logger.info(f"预测完成，结果保存在: {predictions_dir}")
        logger.info(f"类别统计: {all_class_counts}")
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()