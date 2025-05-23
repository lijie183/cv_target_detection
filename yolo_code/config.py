import os
import torch
from pathlib import Path

# 项目根目录
ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据集路径
TRAIN_DATA_PATH = ROOT / "data" / "VOCtrainval_06-Nov-2007" / "VOCdevkit" / "VOC2007"
TEST_DATA_PATH = ROOT / "data" / "VOCtest_06-Nov-2007" / "VOCdevkit" / "VOC2007"

# 结果保存路径
RESULTS_PATH = ROOT / "yolo_code" / "results"
WEIGHTS_PATH = RESULTS_PATH / "weights"
LOG_PATH = RESULTS_PATH / "logs"
VISUALIZATION_PATH = RESULTS_PATH / "visualization"
PREDICTION_PATH = RESULTS_PATH / "predictions"
METRIC_PATH = RESULTS_PATH / "metrics"

# 确保所有输出目录存在
for path in [RESULTS_PATH, WEIGHTS_PATH, LOG_PATH, VISUALIZATION_PATH, PREDICTION_PATH, METRIC_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# 训练参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.937
WARMUP_EPOCHS = 3
IMG_SIZE = 640
MULTI_SCALE = True
USE_MOSAIC = True
USE_MIXUP = True
USE_CBAM = True
USE_EMA = True
USE_SWA = True

# 类别信息
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(VOC_CLASSES)

# 可视化设置
FONT_PATH = str(ROOT / "yolo_code" / "utils" / "simhei.ttf")
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (64, 64, 0), (0, 64, 64), (64, 0, 64), (192, 0, 0), (0, 192, 0)
]

# 验证和测试参数
CONF_THRESH = 0.25
IOU_THRESH = 0.45
USE_TTA = True  # Test-Time Augmentation