import os
import xml.etree.ElementTree as ET
import shutil
import random
import numpy as np
import platform
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib
from datetime import datetime

def setup_matplotlib_fonts(font_path=None):
    """设置matplotlib字体，兼容不同操作系统"""
    system = platform.system()
    
    try:
        if font_path and Path(font_path).exists():
            # 如果指定了字体文件并且存在，优先使用它
            matplotlib.font_manager.fontManager.addfont(font_path)
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
    
    log_file = log_dir / f"prepare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

def create_folder_structure(base_dir):
    """创建所需的文件夹结构"""
    base_path = Path(base_dir)
    
    folders = [
        base_path / "scripts",
        base_path / "pretrained",
        base_path / "config",
        base_path / "results" / "logs",
        base_path / "results" / "models",
        base_path / "results" / "plots",
        base_path / "results" / "metrics",
        base_path / "results" / "predictions",
        base_path / "data" / "labels" / "train",
        base_path / "data" / "labels" / "val",
        base_path / "data" / "images" / "train",
        base_path / "data" / "images" / "val",
    ]
    
    for folder in folders:
        folder.mkdir(exist_ok=True, parents=True)
        print(f"目录创建成功: {folder}")

def get_classes(anno_dir):
    """从标注文件中获取所有类别"""
    classes = set()
    anno_dir = Path(anno_dir)
    
    for xml_file in anno_dir.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes.add(class_name)
        except Exception as e:
            print(f"解析文件 {xml_file} 出错: {e}")
    
    return sorted(list(classes))

def convert_xml_to_yolo(xml_path, output_path, classes):
    """将单个XML文件转换为YOLO格式"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        with open(output_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = classes.index(class_name)
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 转换为YOLO格式 (x_center, y_center, width, height) - 归一化到0-1
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                # 防止超出边界
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                bbox_width = max(0, min(1, bbox_width))
                bbox_height = max(0, min(1, bbox_height))
                
                # 写入标签文件
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
        
        return True
    except Exception as e:
        print(f"转换文件 {xml_path} 出错: {e}")
        return False

def convert_dataset(data_dir, output_dir, dataset_type, classes):
    """转换整个数据集（训练集或验证集）"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    anno_dir = data_dir / 'annotations' / 'xmls'
    img_dir = data_dir / 'images'
    
    output_labels_dir = output_dir / 'labels' / dataset_type
    output_images_dir = output_dir / 'images' / dataset_type
    
    output_labels_dir.mkdir(exist_ok=True, parents=True)
    output_images_dir.mkdir(exist_ok=True, parents=True)
    
    image_files = []
    success_count = 0
    failure_count = 0
    
    # 处理标注文件
    for xml_file in anno_dir.glob('*.xml'):
        # 获取基本文件名（不带扩展名）
        base_name = xml_file.stem
        
        # 查找匹配的图像文件
        img_path = None
        for ext in ['.jpeg', '.jpg', '.png']:
            possible_path = img_dir / f"{base_name}{ext}"
            if possible_path.exists():
                img_path = possible_path
                break
        
        if img_path is None:
            print(f"警告: 未找到与 {xml_file.name} 对应的图像文件")
            failure_count += 1
            continue
            
        # 转换XML到YOLO格式
        output_label_path = output_labels_dir / f"{base_name}.txt"
        success = convert_xml_to_yolo(xml_file, output_label_path, classes)
        
        if not success:
            print(f"转换 {xml_file.name} 失败")
            failure_count += 1
            continue
        
        # 复制图像文件
        img_extension = img_path.suffix
        output_img_path = output_images_dir / f"{base_name}{img_extension}"
        shutil.copy(img_path, output_img_path)
        
        image_files.append(f"{base_name}{img_extension}")
        success_count += 1
    
    print(f"{dataset_type}集转换结果: 成功={success_count}, 失败={failure_count}")
    return image_files

def create_data_yaml(classes, output_dir, use_relative_path=False):
    """创建YAML配置文件"""
    output_dir = Path(output_dir)
    yaml_path = output_dir / "config" / "insects.yaml"
    
    # 训练和验证数据路径
    if use_relative_path:
        train_path = "./data/images/train"
        val_path = "./data/images/val"
    else:
        train_path = str((output_dir / "data" / "images" / "train").resolve())
        val_path = str((output_dir / "data" / "images" / "val").resolve())
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"# 训练和验证数据路径\n")
        f.write(f"train: {train_path}\n")
        f.write(f"val: {val_path}\n\n")
        f.write(f"# 类别数\n")
        f.write(f"nc: {len(classes)}\n\n")
        f.write(f"# 类别名称\n")
        f.write(f"names: {classes}\n")
    
    return yaml_path

def visualize_dataset_distribution(train_dir, val_dir, classes, output_dir):
    """可视化数据集分布情况"""
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    output_dir = Path(output_dir)
    
    train_counts = {cls: 0 for cls in classes}
    val_counts = {cls: 0 for cls in classes}
    
    # 统计训练集中的类别分布
    for txt_file in train_dir.glob('*.txt'):
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(classes):
                            train_counts[classes[class_id]] += 1
                        else:
                            print(f"警告: 训练集文件 {txt_file} 中类别ID {class_id} 超出范围")
        except Exception as e:
            print(f"处理训练集文件 {txt_file} 出错: {e}")
    
    # 统计验证集中的类别分布
    for txt_file in val_dir.glob('*.txt'):
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(classes):
                            val_counts[classes[class_id]] += 1
                        else:
                            print(f"警告: 验证集文件 {txt_file} 中类别ID {class_id} 超出范围")
        except Exception as e:
            print(f"处理验证集文件 {txt_file} 出错: {e}")
    
    # 绘制柱状图
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(classes))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], train_counts.values(), width, label='训练集')
        ax.bar([i + width/2 for i in x], val_counts.values(), width, label='验证集')
        
        ax.set_xlabel('类别名称')
        ax.set_ylabel('样本数量')
        ax.set_title('数据集类别分布')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "results" / "plots" / "class_distribution.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"绘制类别分布图出错: {e}")

def visualize_bounding_boxes(img_dir, label_dir, classes, output_dir, num_samples=5):
    """可视化边界框"""
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    img_files = list(img_dir.glob('*.jp*g')) + list(img_dir.glob('*.png'))
    if len(img_files) > num_samples:
        img_files = random.sample(img_files, num_samples)
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes)))
    
    for img_file in img_files:
        base_name = img_file.stem
        label_file = label_dir / f"{base_name}.txt"
        
        if not label_file.exists():
            print(f"警告: 找不到对应的标签文件: {label_file}")
            continue
        
        try:
            # 读取图像和标注
            img = Image.open(img_file)
            width, height = img.size
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(np.array(img))
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"警告: 标签文件格式不正确: {label_file}")
                        continue
                        
                    class_id = int(parts[0])
                    if class_id >= len(classes):
                        print(f"警告: 类别ID {class_id} 超出范围")
                        continue
                        
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height
                    
                    # 计算坐标
                    xmin = x_center - bbox_width / 2
                    ymin = y_center - bbox_height / 2
                    xmax = x_center + bbox_width / 2
                    ymax = y_center + bbox_height / 2
                    
                    # 绘制边界框
                    rect = plt.Rectangle((xmin, ymin), bbox_width, bbox_height,
                                        linewidth=2, edgecolor=colors[class_id],
                                        facecolor='none')
                    ax.add_patch(rect)
                    
                    # 添加类别标签
                    plt.text(xmin, ymin - 5, classes[class_id], 
                            bbox=dict(facecolor=colors[class_id], alpha=0.5),
                            fontsize=10, color='white')
            
            ax.set_title(f'示例图像: {img_file.name}')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / "results" / "plots" / f"bbox_vis_{base_name}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"可视化图像 {img_file} 出错: {e}")

def main():
    # 获取脚本所在目录并计算相对路径
    script_path = Path(__file__).resolve()
    base_dir = script_path.parents[2]  # 项目根目录 (往上两级)
    data_dir = base_dir / "data" / "insects"
    output_dir = base_dir / "yolo_code"
    
    # 尝试加载本地字体或项目根目录下的字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 设置日志
    logs_dir = output_dir / "results" / "logs"
    logger = setup_logging(logs_dir)
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"系统平台: {platform.system()}")
    
    # 创建文件夹结构
    create_folder_structure(output_dir)
    
    # 获取类别列表
    train_anno_dir = data_dir / "train" / "annotations" / "xmls"
    
    if not train_anno_dir.exists():
        logger.error(f"训练集标注目录不存在: {train_anno_dir}")
        print(f"错误: 训练集标注目录不存在: {train_anno_dir}")
        return
    
    classes = get_classes(train_anno_dir)
    logger.info(f"发现的类别: {classes}")
    print(f"发现的类别: {classes}")
    
    # 转换训练集
    logger.info("转换训练集...")
    print("转换训练集...")
    train_files = convert_dataset(
        data_dir / 'train',
        output_dir / 'data',
        'train',
        classes
    )
    logger.info(f"成功转换 {len(train_files)} 个训练样本")
    print(f"成功转换 {len(train_files)} 个训练样本")
    
    # 转换验证集
    logger.info("转换验证集...")
    print("转换验证集...")
    val_files = convert_dataset(
        data_dir / 'val',
        output_dir / 'data',
        'val',
        classes
    )
    logger.info(f"成功转换 {len(val_files)} 个验证样本")
    print(f"成功转换 {len(val_files)} 个验证样本")
    
    # 创建YAML配置文件 - 使用相对路径增加可移植性
    yaml_path = create_data_yaml(classes, output_dir, use_relative_path=True)
    logger.info(f"配置文件已创建: {yaml_path}")
    print(f"配置文件已创建: {yaml_path}")
    
    # 可视化数据集分布
    logger.info("可视化数据集分布...")
    print("可视化数据集分布...")
    visualize_dataset_distribution(
        output_dir / 'data' / 'labels' / 'train',
        output_dir / 'data' / 'labels' / 'val',
        classes,
        output_dir
    )
    
    # 可视化边界框
    logger.info("可视化部分样本边界框...")
    print("可视化部分样本边界框...")
    visualize_bounding_boxes(
        output_dir / 'data' / 'images' / 'train',
        output_dir / 'data' / 'labels' / 'train',
        classes,
        output_dir,
        num_samples=5
    )
    
    logger.info("数据准备完成!")
    print("数据准备完成!")

if __name__ == "__main__":
    main()