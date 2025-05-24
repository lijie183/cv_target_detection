import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import argparse
from ultralytics.models import YOLO
from tqdm import tqdm
from pathlib import Path
import matplotlib
import platform
import logging
import traceback
import shutil
import glob
# 正确导入datetime
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
    
    log_file = os.path.join(log_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """绘制混淆矩阵"""
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        
        # 设置刻度和标签
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # 添加数值标签
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(output_path, dpi=300)
        plt.close()
        return True
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")
        return False

def plot_pr_curve(precisions, recalls, class_names, output_path):
    """绘制PR曲线"""
    try:
        plt.figure(figsize=(10, 8))
        for i, (precision, recall, name) in enumerate(zip(precisions, recalls, class_names)):
            plt.plot(recall, precision, label=f'{name}')
        
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return True
    except Exception as e:
        print(f"绘制PR曲线失败: {e}")
        return False

def visualize_predictions(model, images_dir, annotations_dir, class_names, output_dir, num_samples=5, conf_threshold=0.25):
    """可视化预测结果"""
    try:
        # 确保目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.jpeg')) + list(Path(images_dir).glob('*.png'))
        
        if not image_files:
            print(f"警告：未在 {images_dir} 找到图像文件")
            return False
        
        # 如果图像数量超过样本数，则随机抽样
        if len(image_files) > num_samples:
            image_files = np.random.choice(image_files, num_samples, replace=False)
        
        # 生成颜色映射 - 修改：不再转换为0-255范围
        colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
        # 保持颜色在0-1范围内，适合matplotlib
        
        error_count = 0
        max_errors = 10
        
        for img_path in image_files:
            base_name = img_path.stem
            
            # 加载图像
            try:
                img = Image.open(img_path)
                width, height = img.size
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
                continue
            
            # 进行预测
            try:
                results = model(img_path, conf=conf_threshold)
            except Exception as e:
                print(f"模型预测失败 {img_path}: {e}")
                continue
            
            plt.figure(figsize=(12, 6))
            
            # 绘制原始图像和标注
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img))
            plt.title('原始标注')
            plt.axis('off')
            
            # 加载真实标注
            annotation_path = Path(annotations_dir) / f"{base_name}.txt"
            if annotation_path.exists():
                try:
                    with open(annotation_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # 确保标注格式正确
                                class_id = int(parts[0])
                                if class_id < len(class_names):  # 确保类别ID有效
                                    x_center = float(parts[1]) * width
                                    y_center = float(parts[2]) * height
                                    bbox_width = float(parts[3]) * width
                                    bbox_height = float(parts[4]) * height
                                    
                                    # 计算坐标
                                    xmin = max(0, x_center - bbox_width / 2)
                                    ymin = max(0, y_center - bbox_height / 2)
                                    xmax = min(width, x_center + bbox_width / 2)
                                    ymax = min(height, y_center + bbox_height / 2)
                                    
                                    # 使用0-1范围的颜色
                                    color = colors[class_id % len(colors)]
                                    
                                    # 绘制边界框
                                    rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                                     linewidth=2, edgecolor=color,
                                                     facecolor='none')
                                    plt.gca().add_patch(rect)
                                    
                                    # 添加类别标签
                                    plt.text(xmin, ymin - 5, class_names[class_id], 
                                         bbox=dict(facecolor=color, alpha=0.5),
                                         fontsize=8, color='white')
                except Exception as e:
                    print(f"无法加载标注 {annotation_path}: {e}")
            
            # 绘制预测结果
            plt.subplot(1, 2, 2)
            plt.imshow(np.array(img))
            plt.title('模型预测')
            plt.axis('off')
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    try:
                        # 获取预测坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取预测类别和置信度
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls_id < len(class_names):  # 确保类别ID有效
                            # 使用0-1范围的颜色
                            color = colors[cls_id % len(colors)]
                            
                            # 绘制边界框
                            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                             linewidth=2, edgecolor=color,
                                             facecolor='none')
                            plt.gca().add_patch(rect)
                            
                            # 添加类别标签和置信度
                            plt.text(x1, y1 - 5, f'{class_names[cls_id]} {conf:.2f}', 
                                   bbox=dict(facecolor=color, alpha=0.5),
                                   fontsize=8, color='white')
                    except Exception as e:
                        error_count += 1
                        if error_count <= max_errors:
                            print(f"处理预测框失败: {e}, cls_id={cls_id}, box值={box.xyxy[0].cpu().numpy()}")
                        elif error_count == max_errors + 1:
                            print("过多预测框错误，后续错误将不再显示...")
                        continue
            
            plt.tight_layout()
            
            # 保存图像
            output_path = Path(output_dir) / f"pred_vs_true_{base_name}.png"
            try:
                plt.savefig(output_path, dpi=300)
                print(f"已保存评估预测可视化: {output_path}")
            except Exception as e:
                print(f"保存图像失败 {output_path}: {e}")
            
            plt.close()
        
        return True
    except Exception as e:
        print(f"可视化预测失败: {e}")
        traceback.print_exc()
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估目标检测模型")
    parser.add_argument("--model", type=str, default="best_model.pt", help="模型文件名")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU阈值")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument("--img-size", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--samples", type=int, default=5, help="可视化样本数量")
    parser.add_argument("--train-dir", type=str, default=None, help="指定训练目录，例如train7")
    return parser.parse_args()

def find_latest_model(base_dir):
    """查找所有可能的模型路径，返回最新的一个"""
    # 模型可能的位置
    possible_locations = [
        # 模型目录中的模型
        list(Path(base_dir).glob('**/models/*.pt')),
        # train目录中的weights目录
        list(Path(base_dir).glob('**/train*/weights/*.pt')),
    ]
    
    all_models = []
    for location in possible_locations:
        all_models.extend(location)
    
    if not all_models:
        return None
    
    # 按修改时间排序，返回最新的
    return sorted(all_models, key=os.path.getmtime)[-1]

def find_model_in_train_dir(base_dir, train_dir_name):
    """在指定的train目录中查找模型"""
    # 查找模式: train7/weights/best.pt 或 train7/weights/last.pt
    train_dir = Path(base_dir) / train_dir_name
    best_model = train_dir / "weights" / "best.pt"
    last_model = train_dir / "weights" / "last.pt"
    
    if best_model.exists():
        return best_model
    elif last_model.exists():
        return last_model
    else:
        # 查找该目录下的任何.pt文件
        models = list(train_dir.glob("**/*.pt"))
        if models:
            return models[0]
    
    return None

def find_latest_train_dir(results_dir):
    """查找数字最大的train目录"""
    train_dirs = []
    for d in results_dir.glob("train*"):
        if d.is_dir():
            # 提取目录名中的数字
            try:
                # 提取目录名中的数字部分 (如 'train11' -> 11)
                num = int(d.name.replace('train', '')) if d.name != 'train' else 0
                train_dirs.append((num, d))
            except ValueError:
                # 如果目录名不符合'trainX'格式，则忽略
                continue
    
    # 按数字大小排序，取最大的
    if train_dirs:
        train_dirs.sort(key=lambda x: x[0], reverse=True)
        return train_dirs[0][1]  # 返回目录路径
    return None

def main():
    args = parse_args()
    
    # 使用Path对象处理路径，更好地兼容Windows和Linux路径
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parents[1]  # 获取项目根目录
    yolo_dir = base_dir / "yolo_code"
    
    # 设置目录
    config_path = yolo_dir / "config" / "insects.yaml"
    results_dir = yolo_dir / "results"
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"
    metrics_dir = results_dir / "metrics"
    logs_dir = results_dir / "logs"
    
    # 创建预测目录
    predictions_base_dir = results_dir / "predictions"
    # 创建evaluate子目录
    evaluate_dir = predictions_base_dir / "evaluate"
    evaluate_dir.mkdir(exist_ok=True, parents=True)
    # 创建带时间戳的评估子目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_eval_dir = evaluate_dir / timestamp
    current_eval_dir.mkdir(exist_ok=True, parents=True)

    # 使用新的输出目录
    predictions_dir = current_eval_dir  # 将当前预测输出指向带时间戳的子目录
    
    # 确保目录存在
    for directory in [metrics_dir, plots_dir, logs_dir, predictions_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # 尝试加载本地字体或项目根目录下的字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 设置日志
    logger = setup_logging(logs_dir)
    logger.info("开始评估目标检测模型")
    logger.info(f"系统环境: {platform.system()} {platform.release()}")
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA是否可用: {cuda_available}")
    if cuda_available:
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        val_data_path = config.get('val', '')
        class_names = config.get('names', [])
        
        logger.info(f"加载数据集配置，类别: {class_names}")
        logger.info(f"验证数据路径: {val_data_path}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return
    
    # 加载模型 - 修改模型加载逻辑
    model_path = None
    
    # 1. 首先在models目录查找最佳模型
    models_best = models_dir / "best_model.pt"
    if models_best.exists():
        model_path = models_best
        logger.info(f"使用models目录中的模型: {model_path}")

    # 2. 如果没有找到，尝试查找数字最大的train目录中的最佳模型
    if not model_path:
        latest_train_dir = find_latest_train_dir(results_dir)
        if latest_train_dir:
            best_model_path = latest_train_dir / "weights" / "best.pt"
            if best_model_path.exists():
                model_path = best_model_path
                logger.info(f"使用最新训练目录({latest_train_dir.name})中的模型: {model_path}")
            else:
                # 尝试last.pt
                last_model_path = latest_train_dir / "weights" / "last.pt"
                if last_model_path.exists():
                    model_path = last_model_path
                    logger.info(f"使用最新训练目录({latest_train_dir.name})中的最后模型: {model_path}")

    # 3. 如果还是没有找到，尝试命令行指定的模型
    if not model_path and args.train_dir:
        train_dir_path = results_dir / args.train_dir
        train_model_path = find_model_in_train_dir(train_dir_path, "")
        if train_model_path:
            model_path = train_model_path
            logger.info(f"使用指定训练目录中的模型: {model_path}")
                
    # 4. 如果还是没有找到，尝试查找整个目录中的任何.pt文件
    if not model_path:
        latest_model = find_latest_model(yolo_dir)
        if latest_model:
            model_path = latest_model
            logger.info(f"使用找到的最新模型: {model_path}")
    
    # 如果所有方法都无法找到模型，则退出
    if not model_path or not model_path.exists():
        logger.error(f"未找到可用的模型文件")
        return
    
    logger.info(f"加载模型: {model_path}")
    try:
        model = YOLO(model_path)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return
    
    # 开始评估
    logger.info("开始评估模型")
    try:
        val_results = model.val(
            data=str(config_path),
            split='val',
            batch=args.batch,
            imgsz=args.img_size,
            conf=args.conf,
            iou=args.iou,
            max_det=300,
            save_json=True,
            save_txt=True,
            plots=True,
            project=str(metrics_dir),
            name='evaluation'
        )
        
        logger.info("评估完成")
        metrics = val_results.box
        
        # 保存评估指标
        metrics_file = metrics_dir / "metrics.json"
        try:
            # 修改为:
            metrics_data = {
                'mAP50': float(metrics.map50),
                'mAP50-95': float(metrics.map)
            }
            # 处理精确率和召回率，它们可能是数组
            if hasattr(metrics.p, 'mean'):
                metrics_data['precision'] = float(metrics.p.mean())
                metrics_data['recall'] = float(metrics.r.mean())
                # 使用平均值计算F1
                metrics_data['f1-score'] = 2 * metrics_data['precision'] * metrics_data['recall'] / (metrics_data['precision'] + metrics_data['recall'] + 1e-8)
            else:
                # 尝试直接转换
                try:
                    metrics_data['precision'] = float(metrics.p)
                    metrics_data['recall'] = float(metrics.r)
                    metrics_data['f1-score'] = 2 * metrics_data['precision'] * metrics_data['recall'] / (metrics_data['precision'] + metrics_data['recall'] + 1e-8)
                except:
                    logger.warning("无法转换精确率/召回率为标量，可能是数组格式")
                    metrics_data['precision'] = float(np.mean(metrics.p)) if isinstance(metrics.p, np.ndarray) else 0
                    metrics_data['recall'] = float(np.mean(metrics.r)) if isinstance(metrics.r, np.ndarray) else 0
                    metrics_data['f1-score'] = 2 * metrics_data['precision'] * metrics_data['recall'] / (metrics_data['precision'] + metrics_data['recall'] + 1e-8)
            
            # 添加每个类别的指标
            class_metrics = {}
            for i, name in enumerate(class_names):
                if i < len(metrics.ap50):
                    try:
                        class_metrics[name] = {
                            'precision': float(metrics.p_per_class[i]) if i < len(metrics.p_per_class) else 0,
                            'recall': float(metrics.r_per_class[i]) if i < len(metrics.r_per_class) else 0,
                            'ap50': float(metrics.ap50[i]),
                            'ap': float(metrics.ap[i]) if i < len(metrics.ap) else 0,
                        }
                    except TypeError:
                        # 处理无法转换为标量的情况
                        logger.warning(f"类别 {name} 的指标无法转换为标量，使用默认值")
                        class_metrics[name] = {
                            'precision': 0.0,
                            'recall': 0.0,
                            'ap50': float(metrics.ap50[i]),
                            'ap': float(metrics.ap[i]) if i < len(metrics.ap) else 0,
                        }
            
            metrics_data['class_metrics'] = class_metrics
            
            # 保存到文件
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)
                
            logger.info(f"评估指标已保存至: {metrics_file}")
            
            # 打印主要指标
            logger.info(f"mAP@0.5: {metrics_data['mAP50']:.4f}")
            logger.info(f"mAP@0.5:0.95: {metrics_data['mAP50-95']:.4f}")
            logger.info(f"精确率: {metrics_data['precision']:.4f}")
            logger.info(f"召回率: {metrics_data['recall']:.4f}")
            logger.info(f"F1分数: {metrics_data['f1-score']:.4f}")
            
            # 打印每个类别的指标
            logger.info("\n各类别性能:")
            for class_name, metrics in class_metrics.items():
                logger.info(f"{class_name}: 精确率={metrics['precision']:.4f}, 召回率={metrics['recall']:.4f}, mAP@0.5={metrics['ap50']:.4f}")
            
        except Exception as e:
            logger.error(f"保存指标失败: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 复制评估生成的图表
        eval_results_dir = metrics_dir / 'evaluation'
        if eval_results_dir.exists():
            for file_path in eval_results_dir.glob('*.png'):
                if file_path.is_file():
                    try:
                        target_path = plots_dir / file_path.name
                        logger.info(f"复制评估图表: {file_path.name}")
                        shutil.copy(file_path, target_path)
                    except Exception as e:
                        logger.error(f"复制文件失败 {file_path}: {str(e)}")
        
        # 可视化预测结果
        logger.info("可视化预测结果")
        val_images_dir = yolo_dir / "data" / "images" / "val"
        val_labels_dir = yolo_dir / "data" / "labels" / "val"
        
        if val_images_dir.exists() and val_labels_dir.exists():
            success = visualize_predictions(
                model,
                val_images_dir,
                val_labels_dir,
                class_names,
                predictions_dir,
                num_samples=args.samples,
                conf_threshold=args.conf
            )
            
            if success:
                logger.info(f"预测可视化已保存至: {predictions_dir}")
                logger.info(f"时间戳目录: {timestamp}")
            else:
                logger.warning("预测可视化生成失败")
        else:
            logger.warning(f"验证数据目录不存在: {val_images_dir} 或 {val_labels_dir}")
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    logger.info("评估完成，所有结果已保存")

if __name__ == "__main__":
    main()