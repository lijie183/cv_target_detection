import os
import shutil
import xml.etree.ElementTree as ET
import random
import numpy as np
import yaml
import cv2
from pathlib import Path
import argparse
from collections import Counter
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import platform
import logging
from datetime import datetime
import matplotlib.font_manager

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
        # 使用更美观的样式
        plt.style.use('ggplot')
        return True
    except Exception as e:
        print(f"字体设置失败: {e}, 可能会导致中文显示异常")
        return False

def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"prepare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def create_directory_structure(base_dir):
    """创建必要的目录结构"""
    directories = [
        "data/images/train",
        "data/images/val",
        "data/images/test",
        "data/annotations/train",
        "data/annotations/val",
        "data/annotations/test",
        "models",
        "results/train",
        "results/predictions",
        "results/evaluation",
        "results/comparison",
        "results/statistics",
        "results/logs",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    
    print(f"创建目录结构完成。基础目录: {base_dir}")

def parse_xml_annotation(xml_file):
    """解析XML注释文件，返回边界框和类名"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    img_path = root.find('path').text if root.find('path') is not None else ""
    img_file = root.find('filename').text
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        # 确保边界框坐标在图像范围内
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'filename': img_file,
        'path': img_path,
        'width': width,
        'height': height,
        'objects': objects
    }

def convert_to_coco_format(annotations, class_names):
    """将注释转换为COCO格式"""
    coco_data = {
        "info": {
            "description": "Insect Detection Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "Custom",
            "date_created": "2023"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "Unknown"
            }
        ],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 添加类别
    for idx, name in enumerate(class_names):
        coco_data["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "insect"
        })
    
    img_id = 0
    anno_id = 0
    
    for img_anno in annotations:
        # 添加图像
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_anno['filename'],
            "width": img_anno['width'],
            "height": img_anno['height']
        })
        
        # 添加注释
        for obj in img_anno['objects']:
            x_min, y_min, x_max, y_max = obj['bbox']
            width = x_max - x_min
            height = y_max - y_min
            
            coco_data["annotations"].append({
                "id": anno_id,
                "image_id": img_id,
                "category_id": class_names.index(obj['name']),
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]
            })
            
            anno_id += 1
        
        img_id += 1
    
    return coco_data

def process_dataset(source_dir, dest_base_dir, split_ratio={'train': 0.7, 'val': 0.2, 'test': 0.1}):
    """处理数据集并划分为训练、验证和测试集"""
    # 创建目录结构
    create_directory_structure(dest_base_dir)
    
    # 获取所有XML文件
    xml_files = []
    source_dir = Path(source_dir)
    
    # 获取类别列表
    train_anno_dir = source_dir / "train" / "annotations" / "xmls"
    if not train_anno_dir.exists():
        print(f"错误: 训练集标注目录不存在: {train_anno_dir}")
        return [], {}

    # 调整查找XML文件的方式，增加对多层目录结构的支持
    for dataset_type in ['train', 'val', 'test']:
        anno_dir = source_dir / dataset_type / "annotations" / "xmls"
        if anno_dir.exists():
            for xml_file in anno_dir.glob('*.xml'):
                xml_files.append((xml_file, dataset_type))
    
    if not xml_files:
        print("错误: 没有找到任何XML标注文件")
        return [], {}
        
    # 统计类别
    all_classes = set()
    annotations = []
    
    print("解析XML文件...")
    for xml_file, dataset_type in tqdm(xml_files):
        try:
            anno = parse_xml_annotation(xml_file)
            
            # 找到对应的图像文件
            img_name = anno['filename']
            img_path = None
            
            # 获取XML文件的目录结构
            xml_dir = xml_file.parent
                
            # 查找对应的图像文件 - 基于目录结构
            img_dir = source_dir / dataset_type / "images"
            
            # 尝试几种可能的扩展名
            for ext in ['.jpg', '.jpeg', '.png']:
                possible_img = img_dir / f"{xml_file.stem}{ext}"
                if possible_img.exists():
                    img_path = possible_img
                    break
            
            if img_path is None:
                print(f"警告: 找不到图像文件 {img_name}，XML路径: {xml_file}")
                continue
                
            # 收集所有类别
            for obj in anno['objects']:
                all_classes.add(obj['name'])
                
            annotations.append({
                'xml_file': xml_file,
                'img_path': img_path,
                'annotation': anno,
                'dataset_type': dataset_type
            })
        except Exception as e:
            print(f"处理文件 {xml_file} 时出错: {e}")
    
    # 排序类别
    class_names = sorted(list(all_classes))
    num_classes = len(class_names)
    print(f"总共找到 {num_classes} 个类别: {class_names}")
    print(f"总共找到 {len(annotations)} 个有效样本")
    
    # 统计每个类别的实例数量
    class_counts = Counter()
    for anno in annotations:
        for obj in anno['annotation']['objects']:
            class_counts[obj['name']] += 1
    
    print("\n类别统计:")
    for cls, count in class_counts.most_common():
        print(f"{cls}: {count} 实例")
    
    # 保存类别信息
    config = {
        'num_classes': num_classes + 1,  # 包括背景类
        'names': class_names,
        'class_counts': {cls: count for cls, count in class_counts.items()}
    }
    
    config_dir = Path(dest_base_dir) / 'config'
    config_dir.mkdir(exist_ok=True, parents=True)
    
    with open(config_dir / 'insects.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    print("类别配置保存到 config/insects.yaml")
    
    # 按照已有的数据集划分处理每个集合
    dataset_splits = {}
    for dataset_type in ['train', 'val', 'test']:
        dataset_splits[dataset_type] = [item for item in annotations if item['dataset_type'] == dataset_type]
    
    # 如果测试集为空，从验证集中分配一部分
    if not dataset_splits['test'] and dataset_splits['val']:
        val_size = len(dataset_splits['val'])
        test_size = int(val_size * 0.3)  # 分配30%的验证集作为测试集
        dataset_splits['test'] = dataset_splits['val'][-test_size:]
        dataset_splits['val'] = dataset_splits['val'][:-test_size]
    
    # 处理每个集合
    for data_split in ['train', 'val', 'test']:
        print(f"\n处理{data_split}集...")
        data = dataset_splits[data_split]
        
        if not data:
            print(f"{data_split}集为空，跳过处理")
            continue
        
        # 收集每个集合的注释
        split_annotations = []
        
        for item in tqdm(data):
            img_file = item['img_path']
            xml_file = item['xml_file']
            anno = item['annotation']
            
            # 目标路径
            dest_img_path = Path(dest_base_dir) / 'data' / 'images' / data_split / img_file.name
            dest_anno_path = Path(dest_base_dir) / 'data' / 'annotations' / data_split / xml_file.name
            
            # 创建目录
            dest_img_path.parent.mkdir(exist_ok=True, parents=True)
            dest_anno_path.parent.mkdir(exist_ok=True, parents=True)
            
            # 复制文件
            try:
                shutil.copy(img_file, dest_img_path)
                shutil.copy(xml_file, dest_anno_path)
                split_annotations.append(anno)
            except Exception as e:
                print(f"复制文件时出错: {e}")
        
        # 转换为COCO格式
        coco_data = convert_to_coco_format(split_annotations, class_names)
        coco_file = Path(dest_base_dir) / 'data' / 'annotations' / f'{data_split}.json'
        
        with open(coco_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=4)
        
        print(f"COCO格式注释保存到 {coco_file}")
    
    print("\n数据准备完成!")
    
    # 可视化类别分布
    plt.figure(figsize=(12, 8))
    classes = list(class_counts.keys())
    values = list(class_counts.values())
    
    # 按值排序
    sorted_indices = np.argsort(values)[::-1]
    classes = [classes[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    plt.bar(classes, values)
    plt.xticks(rotation=45, ha='right')
    plt.title('类别分布')
    plt.xlabel('类别')
    plt.ylabel('实例数量')
    plt.tight_layout()
    
    stats_dir = Path(dest_base_dir) / 'results' / 'statistics'
    stats_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(stats_dir / 'class_distribution.png')
    plt.close()
    
    # 可视化一些示例图像和边界框
    visualize_bounding_boxes(
        Path(dest_base_dir) / 'data' / 'images' / 'train',
        Path(dest_base_dir) / 'data' / 'annotations' / 'train',
        class_names,
        Path(dest_base_dir) / 'results' / 'statistics',
        num_samples=5
    )
    
    return class_names, class_counts

def check_gpu_availability():
    """检查GPU是否可用"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"找到 {gpu_count} 个 GPU:")
        for i, name in enumerate(gpu_names):
            print(f"  GPU {i}: {name}")
        return True, gpu_count, gpu_names
    else:
        print("GPU 不可用，将使用 CPU 进行训练（这可能会很慢）")
        return False, 0, []

def print_directory_structure(path, level=0, max_level=3, max_files=5):
    """打印目录结构，用于调试"""
    if level > max_level:
        return
    
    path = Path(path)
    if not path.exists():
        print(f"路径不存在: {path}")
        return
    
    print('    ' * level + str(path.name) + '/')
    
    try:
        items = list(path.iterdir())
        files = [item for item in items if item.is_file()]
        dirs = [item for item in items if item.is_dir()]
        
        # 显示文件
        for i, file in enumerate(files):
            if i < max_files:
                print('    ' * (level + 1) + str(file.name))
            elif i == max_files:
                print('    ' * (level + 1) + f"... (还有 {len(files) - max_files} 个文件)")
                break
        
        # 递归显示目录
        for directory in dirs:
            print_directory_structure(directory, level + 1, max_level, max_files)
    
    except Exception as e:
        print(f"读取目录 {path} 出错: {e}")

def visualize_bounding_boxes(img_dir, anno_dir, class_names, output_dir, num_samples=5):
    """可视化边界框"""
    img_dir = Path(img_dir)
    anno_dir = Path(anno_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 获取所有XML文件
    xml_files = list(anno_dir.glob('*.xml'))
    if len(xml_files) > num_samples:
        xml_files = random.sample(xml_files, num_samples)
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    for xml_file in xml_files:
        try:
            # 获取对应的图像文件
            base_name = xml_file.stem
            img_file = None
            
            for ext in ['.jpg', '.jpeg', '.png']:
                possible_img = img_dir / f"{base_name}{ext}"
                if possible_img.exists():
                    img_file = possible_img
                    break
            
            if img_file is None:
                print(f"警告: 未找到与 {xml_file.name} 对应的图像文件")
                continue
            
            # 解析XML标注
            anno = parse_xml_annotation(xml_file)
            
            # 读取图像
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 创建图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img)
            
            # 绘制边界框
            for obj in anno['objects']:
                class_name = obj['name']
                if class_name not in class_names:
                    continue
                
                class_id = class_names.index(class_name)
                xmin, ymin, xmax, ymax = obj['bbox']
                
                # 绘制边界框
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth=2, edgecolor=colors[class_id],
                                    facecolor='none')
                ax.add_patch(rect)
                
                # 添加类别标签
                plt.text(xmin, ymin - 5, class_name, 
                        bbox=dict(facecolor=colors[class_id], alpha=0.5),
                        fontsize=10, color='white')
            
            ax.set_title(f'示例图像: {img_file.name}')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f"bbox_vis_{base_name}.png", dpi=300)
            plt.close()
        
        except Exception as e:
            print(f"可视化 {xml_file} 出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument('--source', type=str, required=False, default='data/insects', help='源数据集路径')
    parser.add_argument('--dest', type=str, required=False, default='frcnn_code', help='目标目录')
    parser.add_argument('--train', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--check_dirs', action='store_true', help='检查并打印目录结构')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取脚本所在目录并计算相对路径
    script_path = Path(__file__).resolve()
    base_dir = script_path.parents[2]  # 项目根目录 (往上两级)
    
    # 设置数据目录和输出目录
    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = base_dir / args.source
    
    dest_dir = Path(args.dest)
    if not dest_dir.is_absolute():
        dest_dir = base_dir / args.dest
    
    # 尝试加载本地字体
    font_path = base_dir / "SimHei.ttf"
    if not font_path.exists():
        font_path = None
    setup_matplotlib_fonts(font_path)
    
    # 设置日志
    logs_dir = dest_dir / "results" / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(logs_dir)
    
    logger.info(f"数据目录: {source_dir}")
    logger.info(f"输出目录: {dest_dir}")
    logger.info(f"系统平台: {platform.system()}")
    
    # 检查目录结构
    if args.check_dirs:
        logger.info("检查数据集目录结构:")
        print("\n检查数据集目录结构:")
        print_directory_structure(source_dir)
    
    # 检查GPU可用性
    has_gpu, gpu_count, gpu_names = check_gpu_availability()
    logger.info(f"GPU可用性: {has_gpu}, 数量: {gpu_count}, 名称: {gpu_names}")
    
    # 处理数据集
    split_ratio = {'train': args.train, 'val': args.val, 'test': 1 - args.train - args.val}
    class_names, class_counts = process_dataset(source_dir, dest_dir, split_ratio)
    
    logger.info(f"类别统计: {class_counts}")
    logger.info("数据准备完成!")
    print("数据准备完成!")

if __name__ == "__main__":
    main()