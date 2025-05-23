import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import platform
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics.models import YOLO
import logging
import shutil
from pathlib import Path
import traceback

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
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 添加编码参数
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def plot_training_progress(metrics, output_dir):
    """绘制训练进度图表"""
    epochs = list(range(1, len(metrics['train_loss']) + 1))
    
    # 损失函数图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    # mAP图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics['val_map50'], 'g-', label='mAP@0.5')
    plt.plot(epochs, metrics['val_map'], 'c-', label='mAP@0.5:0.95')
    plt.title('验证集 mAP')
    plt.xlabel('训练轮次')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_curve.png'), dpi=300)
    plt.close()
    
    # 精确度和召回率
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics['val_precision'], 'm-', label='精确度')
    plt.plot(epochs, metrics['val_recall'], 'y-', label='召回率')
    plt.title('验证集精确度和召回率')
    plt.xlabel('训练轮次')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()

def get_device_info():
    """获取系统环境和设备信息"""
    device_info = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if device_info["cuda_available"] and device_info["gpu_count"] > 0:
        device_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(device_info["gpu_count"])]
    
    return device_info

def set_resource_limits(gpu_id=None, memory_limit=None):
    """设置资源限制"""
    if gpu_id is not None and gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"设置使用GPU: {gpu_id}")
    
    if memory_limit and torch.cuda.is_available():
        # 限制GPU显存使用
        try:
            torch.cuda.set_per_process_memory_fraction(memory_limit)
            print(f"限制GPU内存使用比例: {memory_limit}")
        except:
            print("无法限制GPU内存使用")
    
    # 可选: 设置CPU线程数
    try:
        torch.set_num_threads(4)  # 调整为适当的数值
    except:
        pass

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练目标检测模型")
    parser.add_argument("--batch", type=int, default=16, help="批量大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--patience", type=int, default=10, help="早停轮数")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1表示使用CPU")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--img-size", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--use-relative-path", action="store_true", help="使用相对路径")
    parser.add_argument("--memory-limit", type=float, default=None, help="GPU显存使用限制(0.0-1.0)")
    parser.add_argument("--resume", type=str, default=None, help="断点续训模型路径")
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--model", type=str, default="yolov10n.pt", help="模型类型：yolov10n、yolov10s等")
    parser.add_argument("--cos-lr", action="store_true", help="使用余弦学习率调度")
    return parser.parse_args()

def train_with_resume(model, yaml_path, checkpoint_path=None, device="0", **kwargs):
    """支持断点续训的训练函数"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从检查点恢复训练: {checkpoint_path}")
        try:
            # 尝试从检查点恢复
            results = model.train(
                data=yaml_path,
                resume=checkpoint_path,
                device=device,
                **kwargs
            )
            return results
        except Exception as e:
            print(f"恢复训练失败: {e}")
            print("将从头开始训练...")
    
    # 从头开始训练
    results = model.train(
        data=yaml_path,
        device=device,
        **kwargs
    )
    return results

def find_best_model(results_dir):
    """查找最佳模型路径"""
    weights_dir = Path(results_dir) / "weights"
    best_model = weights_dir / "best.pt"
    
    if best_model.exists():
        return best_model
    
    # 如果找不到best.pt，尝试找最后一个checkpoint
    checkpoints = list(weights_dir.glob("*.pt"))
    if checkpoints:
        return sorted(checkpoints, key=lambda x: os.path.getmtime(x))[-1]
    
    return None

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置资源限制
    set_resource_limits(gpu_id=args.gpu, memory_limit=args.memory_limit)
    
    # 使用Path对象处理路径，更好地兼容Windows路径
    script_dir = Path(__file__).resolve().parent
    
    # 获取项目根目录（向上两级）
    base_dir = script_dir.parents[1]
    
    # 构建相关路径
    yolo_dir = base_dir / "yolo_code"
    config_path = yolo_dir / "config" / "insects.yaml"
    results_dir = yolo_dir / "results"
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"
    logs_dir = results_dir / "logs"
    
    # 创建必要的目录
    models_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 尝试加载本地字体或项目根目录下的字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 设置日志
    logger = setup_logging(logs_dir)
    logger.info("开始训练目标检测模型")
    logger.info(f"系统环境信息: {get_device_info()}")
    
    # 检查数据目录是否存在
    train_dir = yolo_dir / "data" / "images" / "train"
    val_dir = yolo_dir / "data" / "images" / "val"
    
    # 检查关键路径和文件
    logger.info(f"检查关键目录和文件:")
    logger.info(f"训练数据目录: {train_dir}, 存在: {train_dir.exists()}")
    logger.info(f"验证数据目录: {val_dir}, 存在: {val_dir.exists()}")
    logger.info(f"配置文件: {config_path}, 存在: {config_path.exists()}")
    
    if not train_dir.exists() or not val_dir.exists():
        logger.error(f"训练或验证数据集目录不存在！")
        logger.info(f"请先运行 01_prepare_data.py 脚本准备数据")
        return
    
    # 打印训练数据目录中的文件数量
    if train_dir.exists():
        train_files = list(train_dir.glob('*'))
        logger.info(f"训练目录中有 {len(train_files)} 个文件")
        if len(train_files) > 0:
            logger.info(f"示例文件: {[f.name for f in train_files[:3]]}")

    # 打印验证数据目录中的文件数量
    if val_dir.exists():
        val_files = list(val_dir.glob('*'))
        logger.info(f"验证目录中有 {len(val_files)} 个文件")
        if len(val_files) > 0:
            logger.info(f"示例文件: {[f.name for f in val_files[:3]]}")
    
    # 确保标签目录存在
    train_label_dir = yolo_dir / "data" / "labels" / "train"
    val_label_dir = yolo_dir / "data" / "labels" / "val"
    logger.info(f"训练标签目录: {train_label_dir}, 存在: {train_label_dir.exists()}")
    logger.info(f"验证标签目录: {val_label_dir}, 存在: {val_label_dir.exists()}")
    
    # 检查标签文件数量
    if train_label_dir.exists():
        train_label_files = list(train_label_dir.glob('*.txt'))
        logger.info(f"训练标签文件数量: {len(train_label_files)}")
    if val_label_dir.exists():
        val_label_files = list(val_label_dir.glob('*.txt'))
        logger.info(f"验证标签文件数量: {len(val_label_files)}")
    
    # 修改配置文件，根据参数选择相对路径或绝对路径
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 根据命令行参数决定使用相对路径还是绝对路径
    if args.use_relative_path:
        # 使用相对路径
        config['train'] = "./data/images/train"
        config['val'] = "./data/images/val"
        logger.info(f"使用相对路径配置YAML")
    else:
        # 使用绝对路径
        config['train'] = str(train_dir)
        config['val'] = str(val_dir)
        logger.info(f"使用绝对路径配置YAML")
    
    # 保存修改后的配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    logger.info(f"配置文件已更新")
    
    num_classes = config['nc']
    class_names = config['names']
    
    logger.info(f"加载数据集配置: {num_classes}个类别: {class_names}")
    
    # 初始化模型
    try:
        logger.info(f"初始化{args.model}模型...")
        model = YOLO(args.model)  # 使用指定的预训练模型
        logger.info("模型加载成功")
    except Exception as e:
        # 如果加载失败，尝试下载模型
        logger.warning(f"无法加载预训练模型: {str(e)}")
        logger.info(f"尝试下载预训练模型: {args.model}...")
        model = YOLO(args.model)
    
    # 设置训练参数
    batch_size = args.batch
    epochs = args.epochs
    patience = args.patience  # 早停轮数
    img_size = args.img_size
    lr = args.lr
    
    # 训练模型
    logger.info(f"开始训练，最大轮次: {epochs}，早停轮数: {patience}, 批量大小: {batch_size}")
    
    try:
        # 使用YAML配置文件路径而不是数据字典
        logger.info(f"使用配置文件: {config_path}")
        
        # 设置工作目录为项目根目录
        os.chdir(str(base_dir))
        logger.info(f"已将工作目录设置为: {os.getcwd()}")
        
        # 使用配置文件的绝对路径
        yaml_path = str(config_path.resolve())
        logger.info(f"配置文件绝对路径: {yaml_path}")
        
        # 设置设备
        device = str(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 检查是否进行断点续训
        checkpoint_path = args.resume
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                checkpoint_path = models_dir / "last_model.pt"
                if checkpoint_path.exists():
                    logger.info(f"找到上次训练的模型: {checkpoint_path}")
                else:
                    checkpoint_path = None
                    logger.info(f"未找到断点续训模型，将从头开始训练")

        # 准备训练参数
        train_params = {
            'epochs': epochs,
            'patience': patience,
            'batch': batch_size,
            'imgsz': img_size,
            'project': str(results_dir),
            'name': 'train',
            'save': True,
            'save_period': 1,
            'pretrained': True,
            'optimizer': 'AdamW',  # 使用AdamW优化器
            'lr0': lr,          # 初始学习率
            'weight_decay': 0.01,  # 权重衰减
            'box': 7.5,            # box损失权重
            'cls': 0.5,            # 类别损失权重
            'dfl': 1.5,            # DFL损失权重
            'plots': True,         # 保存结果图
            'save_json': True,     # 保存结果为JSON
            'val': True,           # 每轮进行验证
            'deterministic': True, # 确保结果可复现
            'cos_lr': args.cos_lr,  # 使用余弦学习率调度
            'mosaic': 0 if args.no_augment else 1.0,  # 马赛克增强
            'augment': not args.no_augment,  # 数据增强
        }
        
        # 训练模型（支持断点续训）
        train_results = train_with_resume(
            model, 
            yaml_path, 
            checkpoint_path=checkpoint_path,
            device=device,
            **train_params
        )
        
        logger.info("训练完成!")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        # 避免使用可能包含Emoji的traceback
        logger.error("详细错误信息:")
        for line in traceback.format_exc().splitlines():
            logger.error(line)
        return
    
    try:
        # 寻找训练结果目录和最佳模型
        train_output_dir = results_dir / "train"
        
        # 如果没有指定，则寻找包含最新子文件夹的路径，如 "train2", "train3" 等
        if not train_output_dir.exists():
            train_dirs = sorted([d for d in results_dir.glob("train*") if d.is_dir()], 
                               key=lambda x: os.path.getmtime(x))
            if train_dirs:
                train_output_dir = train_dirs[-1]  # 取最新的目录
        
        best_model_path = train_output_dir / "weights" / "best.pt"
        last_model_path = train_output_dir / "weights" / "last.pt"
        
        logger.info(f"训练输出目录: {train_output_dir}")
        
        # 保存最佳模型
        if best_model_path.exists():
            best_model_target = models_dir / 'best_model.pt'
            
            logger.info(f"复制最佳模型 {best_model_path} 到 {best_model_target}")
            shutil.copy(best_model_path, best_model_target)
        else:
            logger.warning(f"未找到最佳模型: {best_model_path}")
        
        # 保存最后一个模型
        if last_model_path.exists():
            last_model_target = models_dir / 'last_model.pt'
            
            logger.info(f"复制最后模型 {last_model_path} 到 {last_model_target}")
            shutil.copy(last_model_path, last_model_target)
        else:
            logger.warning(f"未找到最后训练的模型: {last_model_path}")
        
        # 复制训练生成的图表到plots目录
        if train_output_dir.exists():
            for file in os.listdir(train_output_dir):
                file_path = train_output_dir / file
                if file.endswith(('.png', '.jpg')) and file_path.is_file():
                    try:
                        shutil.copy(file_path, plots_dir / file)
                        logger.info(f"复制图表: {file}")
                    except Exception as e:
                        logger.warning(f"无法复制文件 {file}: {str(e)}")
        
        logger.info("所有训练结果和模型已保存")
        
        # 打印最终模型性能
        if best_model_path.exists():
            try:
                logger.info("验证最佳模型性能:")
                best_model = YOLO(best_model_path)
                val_results = best_model.val(data=yaml_path)
                
                # 打印每个类别的精确度和召回率
                logger.info("\n最佳模型性能指标:")
                logger.info(f"mAP@0.5: {val_results.box.map50:.4f}")
                logger.info(f"mAP@0.5-0.95: {val_results.box.map:.4f}")
                logger.info(f"精确度: {val_results.box.p:.4f}")
                logger.info(f"召回率: {val_results.box.r:.4f}")
                
                # 打印每个类别的性能
                logger.info("\n各类别性能:")
                for i, name in enumerate(class_names):
                    try:
                        class_p = val_results.box.p_curve[i]
                        class_r = val_results.box.r_curve[i]
                        class_map = val_results.box.ap50[i]
                        logger.info(f"{name}: 精确度={class_p:.4f}, 召回率={class_r:.4f}, mAP@0.5={class_map:.4f}")
                    except:
                        logger.info(f"{name}: 无法获取详细性能指标")
            except Exception as e:
                logger.error(f"验证模型性能时出错: {str(e)}")
        
    except Exception as e:
        logger.error(f"保存模型和结果时发生错误: {str(e)}")
        # 避免使用可能包含Emoji的traceback
        logger.error("详细错误信息:")
        for line in traceback.format_exc().splitlines():
            logger.error(line)

if __name__ == "__main__":
    main()