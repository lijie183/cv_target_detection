import torch
import math

class ModelEMA:
    """指数移动平均模型"""
    def __init__(self, model, decay=0.9999, updates=0):
        """
        创建EMA模型
        
        参数:
            model: 需要创建EMA的模型
            decay: EMA衰减率
            updates: 已经进行的更新次数
        """
        # 评估模式
        self.ema = model.eval().clone() if hasattr(model, 'clone') else copy_model(model).eval()
        self.updates = updates  # 更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 指数移动平均的衰减函数
        
        # 更新模型的buffer
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        """
        更新EMA模型
        
        参数:
            model: 当前训练的模型
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            # 更新参数
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
    
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        更新模型属性 (例如 anchors 和 stride等)
        
        参数:
            model: 源模型
            include: 包含的属性列表
            exclude: 排除的属性列表
        """
        # 只需要更新属性而不是参数
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)

def copy_model(model):
    """
    创建模型的深拷贝副本
    
    参数:
        model: 需要拷贝的源模型
    
    返回:
        新的模型副本
    """
    import copy
    return copy.deepcopy(model)


    import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import *

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=True, CIoU=False, eps=1e-7):
    """计算边界框IOU"""
    # 如果需要将中心宽高坐标格式转换为边角坐标格式
    if not x1y1x2y2:
        # 转换box1
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        # 转换box2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # 获取边角坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 交集区域
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 并集区域
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    
    if GIoU or DIoU or CIoU:
        # 最小外接矩形
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 包围盒宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 包围盒高度
        
        if CIoU or DIoU:  # 距离IoU
            c2 = cw ** 2 + ch ** 2 + eps  # 最小外接矩形对角线平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                   (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 中心点距离平方
            
            if DIoU:
                return iou - rho2 / c2  # DIoU
            
            elif CIoU:  # 完整版CIoU
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        
        else:  # GIoU
            c_area = cw * ch + eps  # 最小外接矩形面积
            return iou - (c_area - union) / c_area  # GIoU
    
    else:
        return iou  # 普通IoU

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, device):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        
        # 三个尺度的锚点和对应的步长
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],  # P3/8
            [[30, 61], [62, 45], [59, 119]],  # P4/16
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ]).float().to(device)
        
        self.strides = torch.tensor([8, 16, 32]).to(device)
        
        # 损失函数权重
        self.lambda_coord = 5.0  # 坐标损失权重
        self.lambda_conf = 1.0   # 置信度损失权重
        self.lambda_cls = 1.0    # 分类损失权重
        
        # 用于焦点损失
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
        
        # CIoU Loss
        self.box_loss_fn = bbox_iou
    
    def forward(self, predictions, targets):
        # 初始化损失
        loss_box = torch.zeros(1, device=self.device)    # 边界框回归损失
        loss_obj = torch.zeros(1, device=self.device)    # 目标置信度损失
        loss_cls = torch.zeros(1, device=self.device)    # 分类损失
        
        # 三个尺度的输出
        for i, pred in enumerate(predictions):
            # 获取当前尺度的锚点和步长
            anchors = self.anchors[i]
            stride = self.strides[i]
            
            # 获取特征图尺寸
            bs, _, ny, nx = pred.shape
            
            # 重塑预测结果
            pred = pred.view(bs, 3, 5 + self.num_classes, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # 预测坐标、置信度和分类分数
            pred_xy = torch.sigmoid(pred[..., 0:2])   # 中心点坐标
            pred_wh = torch.exp(pred[..., 2:4]) * anchors.view(1, 3, 1, 1, 2) / stride  # 宽高
            pred_obj = torch.sigmoid(pred[..., 4])    # 置信度
            pred_cls = torch.sigmoid(pred[..., 5:])   # 类别
            
            # 创建网格
            grid_y, grid_x = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid = torch.stack((grid_x, grid_y), 2).view(1, 1, ny, nx, 2).float().to(self.device)
            
            # 计算预测框的绝对坐标
            pred_boxes = torch.cat((pred_xy + grid, pred_wh), dim=4)
            
            # 匹配预测和目标
            for b, target in enumerate(targets):
                # 忽略没有目标的图像
                if len(target['boxes']) == 0:
                    continue
                
                # 目标框和标签
                tcls = target['labels']  # 类别
                tbox = target['boxes']   # 边界框 (x_center, y_center, width, height)
                
                # 计算目标在特征图上的尺度
                tbox_scaled = tbox.clone().to(self.device)
                tbox_scaled[:, 0:2] = tbox_scaled[:, 0:2] * nx  # 中心点坐标
                tbox_scaled[:, 2:4] = tbox_scaled[:, 2:4] * nx  # 宽高
                
                # 为每个目标找到最佳的锚点和网格位置
                for t in range(tbox.size(0)):
                    # 目标框
                    gx, gy = tbox_scaled[t, 0], tbox_scaled[t, 1]  # 网格坐标
                    gw, gh = tbox_scaled[t, 2], tbox_scaled[t, 3]  # 宽高
                    
                    # 网格索引
                    gi, gj = int(gx), int(gy)
                    
                    # 限制索引范围
                    gi = max(0, min(gi, nx-1))
                    gj = max(0, min(gj, ny-1))
                    
                    # 找到最佳锚点
                    anchors_wh = anchors / stride
                    box_wh = torch.tensor([gw, gh]).to(self.device)
                    
                    # 计算与所有锚点的宽高比
                    ratios = box_wh / anchors_wh
                    ratios = torch.max(ratios, 1/ratios)
                    max_ratio, _ = ratios.max(dim=1)
                    
                    # 选择最佳锚点
                    a = torch.argmin(max_ratio)
                    
                    # 目标值
                    target_obj = torch.zeros_like(pred_obj[b])
                    target_obj[a, gj, gi] = 1
                    
                    # 类别
                    target_cls = torch.zeros_like(pred_cls[b])
                    target_cls[a, gj, gi, tcls[t]] = 1
                    
                    # 边界框坐标
                    target_box = torch.zeros_like(pred_boxes[b, a, gj, gi])
                    target_box[0] = gx - gi
                    target_box[1] = gy - gj
                    target_box[2] = gw
                    target_box[3] = gh
                    
                    # 计算边界框损失
                    iou = self.box_loss_fn(
                        pred_boxes[b, a, gj, gi].unsqueeze(0),
                        target_box.unsqueeze(0),
                        x1y1x2y2=False,
                        DIoU=True
                    )
                    loss_box += (1.0 - iou) * self.lambda_coord
                    
                    # 置信度损失
                    loss_obj += self.BCEobj(
                        pred_obj[b, a, gj, gi],
                        target_obj[a, gj, gi]
                    ) * self.lambda_conf
                    
                    # 分类损失
                    loss_cls += self.BCEcls(
                        pred_cls[b, a, gj, gi],
                        target_cls[a, gj, gi]
                    ) * self.lambda_cls
        
        # 总损失
        loss = loss_box + loss_obj + loss_cls
        
        return loss

        import torch
import numpy as np
import torchvision
from scipy.interpolate import interp1d

def xywh2xyxy(x):
    """将(x,y,w,h)格式转换为(x1,y1,x2,y2)格式"""
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def box_iou(box1, box2):
    """计算两组框的IoU"""
    # 获取维度
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    
    # 计算交集区域
    x1 = torch.max(b1_x1[:, None], b2_x1)
    y1 = torch.max(b1_y1[:, None], b2_y1)
    x2 = torch.min(b1_x2[:, None], b2_x2)
    y2 = torch.min(b1_y2[:, None], b2_y2)
    
    # 计算交集面积
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # 计算并集面积
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = box1_area[:, None] + box2_area - intersection
    
    # 计算IoU
    iou = intersection / union
    
    return iou

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """对检测结果进行非极大值抑制"""
    # 检查
    assert 0 <= conf_thres <= 1, f'置信度阈值 {conf_thres} 必须在0-1之间'
    assert 0 <= iou_thres <= 1, f'IoU {iou_thres} 必须在0-1之间'

    # 将预测转换为 [x1, y1, x2, y2, conf, cls]
    nc = prediction.shape[2] - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选对象

    # 设置
    min_wh, max_wh = 2, 4096  # (像素) 最小和最大框宽高
    max_nms = 30000  # NMS前的最大框数
    time_limit = 10.0  # NMS时间限制(秒)
    multi_label &= nc > 1  # 多标签每框(需要 nc > 1)
    
    output = [None] * prediction.shape[0]
    
    for xi, x in enumerate(prediction):  # 图像索引, 图像推理
        # 应用约束条件
        x = x[xc[xi]]  # 保留置信度较高的部分
        
        # 如果没有框，继续
        if not x.shape[0]:
            continue
            
        # 将conf乘以class conf得到conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # 将(center_x, center_y, width, height)转换为(x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # 应用NMS
        if multi_label:
            # 多标签的情况下，对每个类别分别应用NMS
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            # 最佳类别
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # 过滤掉小框
        if not x.shape[0]:
            continue
        
        # 按照置信度排序
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # 基于IoU进行NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        boxes, scores = x[:, :4] + c, x[:, 4]  # 框(偏移cls)，分数
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # 限制检测数
            i = i[:max_det]
        
        output[xi] = x[i]
    
    return output

def compute_ap(pred_boxes, target_boxes, iou_thres=0.5, class_names=None):
    """
    计算平均精度 (AP) 和 mAP
    
    参数:
        pred_boxes (list): 预测框列表
        target_boxes (list): 目标框列表
        iou_thres (float): IoU阈值
        class_names (list): 类别名称列表
    
    返回:
        float: mAP值
    """
    # 如果没有预测或目标，返回0
    if not pred_boxes or not target_boxes:
        return 0.0
    
    # 转换为张量
    if isinstance(pred_boxes[0], torch.Tensor):
        pred_boxes = [p.cpu().numpy() for p in pred_boxes]
    if isinstance(target_boxes[0], torch.Tensor):
        target_boxes = [t.cpu().numpy() for t in target_boxes]
    
    # 计算每个类别的AP
    ap = []
    
    # 如果没有类别名称，使用索引
    if class_names is None:
        class_names = list(range(len(pred_boxes[0])))
    
    for c in range(len(class_names)):
        # 收集当前类别的预测和目标
        p = [box for box in pred_boxes if box[c] > 0]
        t = [box for box in target_boxes if box[c] > 0]
        
        # 如果该类别没有目标，跳过
        if len(t) == 0:
            continue
        
        # 如果该类别没有预测，AP为0
        if len(p) == 0:
            ap.append(0)
            continue
        
        # 按置信度排序
        p.sort(key=lambda x: x[4], reverse=True)
        
        # 初始化TP和FP计数
        tp = np.zeros(len(p))
        fp = np.zeros(len(p))
        
        # 计算TP和FP
        for i, pred in enumerate(p):
            best_iou = 0
            best_idx = -1
            
            for j, target in enumerate(t):
                iou = compute_iou(pred[:4], target[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_iou >= iou_thres:
                tp[i] = 1
                # 删除已匹配的目标
                t.pop(best_idx)
            else:
                fp[i] = 1
        
        # 累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精确率和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / (len(target_boxes) + 1e-10)
        
        # 添加起点和终点
        precision = np.concatenate(([0], precision, [0]))
        recall = np.concatenate(([0], recall, [1]))
        
        # 确保精确率是降序的
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # 计算AP (AUC)
        indices = np.where(recall[1:] != recall[:-1])[0]
        ap.append(np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]))
    
    # 计算mAP
    return np.mean(ap) if len(ap) > 0 else 0.0

def compute_iou(box1, box2):
    """计算两个框的IoU"""
    # 确保box1和box2是numpy数组
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def ap_per_class(tp, conf, pred_cls, target_cls, target_boxes, iou_thres=0.5, eps=1e-16):
    """
    计算每个类别的平均精度 (AP) 和Precision-Recall曲线
    
    参数:
        tp: 预测框
        conf: 置信度分数
        pred_cls: 预测类别
        target_cls: 目标类别
        target_boxes: 目标框
        iou_thres: IoU阈值
    
    返回:
        ap: 平均精度
        p: 精确率
        r: 召回率
    """
    # 找到每个类别的索引
    unique_classes = np.unique(target_cls)
    nc = len(unique_classes)  # 类别数量
    
    # 初始化返回值
    ap = np.zeros((nc))
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    
    # 对每个类别计算AP
    for ci, c in enumerate(unique_classes):
        # 筛选当前类别的预测和目标
        i = pred_cls == c
        n_pred = i.sum()  # 预测数量
        
        i_target = target_cls == c
        n_target = i_target.sum()  # 目标数量
        
        if n_pred == 0 or n_target == 0:
            continue
        
        # 计算IoU
        iou = box_iou(tp[i], target_boxes[i_target])
        
        # 分配预测框到目标框
        matched_indices = torch.where(iou > iou_thres)
        
        # 初始化TP和FP数组
        tpc = np.zeros((n_pred))
        fpc = np.zeros((n_pred))
        
        # 设置已匹配的预测为TP，未匹配的为FP
        if matched_indices[0].shape[0]:
            matches = torch.cat((matched_indices[0].unsqueeze(1), matched_indices[1].unsqueeze(1)), 1)
            
            if len(matches) > 0:
                matches = matches[matches[:, 0].argsort()]
                matches = matches[matches[:, 1].argsort(kind='mergesort')]
                
                # 根据置信度重新排序
                matches = matches[np.argsort(-conf[i][matches[:, 0]])]
                
                # 只保留一对一的匹配
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                
                tpc[matches[:, 0]] = 1
                fpc[~np.isin(np.arange(n_pred), matches[:, 0])] = 1
        
        # 累积TP和FP
        tp_cumsum = np.cumsum(tpc)
        fp_cumsum = np.cumsum(fpc)
        
        # 计算精确率和召回率
        rc = tp_cumsum / (n_target + eps)
        pc = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        
        # 使用VOC07评估方法计算AP
        mrec = np.concatenate(([0.], rc, [1.]))
        mpre = np.concatenate(([0.], pc, [0.]))
        
        # 计算精确率包络
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        
        # 寻找召回率变化点的索引
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # 计算AUC (VOC07方法)
        ap[ci] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        # 插值1000个点用于绘图
        interp = interp1d(np.arange(len(rc)) / (len(rc) - 1), rc)
        r[ci] = interp(np.linspace(0, 1, 1000))
        interp = interp1d(rc, pc)
        p[ci] = interp(r[ci])
    
    return ap, p, r

    import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from matplotlib.font_manager import FontProperties
import torchvision.transforms as T

from config import *

def visualize_batch(image, target, img_id, epoch, is_validation=False):
    """可视化图像和目标框"""
    # 转换图像为numpy数组用于matplotlib显示
    img_np = image.permute(1, 2, 0).numpy()
    
    # 对图像进行标准化还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    # 创建图像
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # 显示图像
    ax.imshow(img_np)
    
    # 获取图像尺寸
    height, width = img_np.shape[:2]
    
    # 在图像上绘制边界框
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            # 将YOLO格式的框 (x_center, y_center, width, height) 转换为 (x_min, y_min, width, height)
            x_center, y_center, box_width, box_height = box.numpy()
            
            # 将归一化坐标转换为像素坐标
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            
            # 计算左上角坐标
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            
            # 获取类别颜色
            color = COLOR_PALETTE[int(label) % len(COLOR_PALETTE)]
            color = [c/255 for c in color]  # 转换为matplotlib需要的0-1范围
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x_min, y_min), box_width, box_height, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加类别标签
            class_name = VOC_CLASSES[int(label)]
            ax.text(
                x_min, y_min - 5, class_name, 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # 添加图像ID和epoch信息
    title = f"{'Validation' if is_validation else 'Training'} - Epoch {epoch+1} - {img_id}"
    ax.set_title(title)
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 设置紧凑布局
    plt.tight_layout()
    
    return fig

def visualize_predictions(image, predictions, target, img_id, class_names, conf_threshold=0.5):
    """可视化检测结果"""
    # 转换图像为numpy数组用于matplotlib显示
    img_np = image.permute(1, 2, 0).numpy()
    
    # 对图像进行标准化还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 显示图像和真实标注
    ax1.imshow(img_np)
    ax1.set_title(f"Ground Truth - {img_id}")
    
    # 获取图像尺寸
    height, width = img_np.shape[:2]
    
    # 在图像上绘制真实边界框
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            # 将YOLO格式的框 (x_center, y_center, width, height) 转换为 (x_min, y_min, width, height)
            x_center, y_center, box_width, box_height = box.numpy()
            
            # 将归一化坐标转换为像素坐标
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            
            # 计算左上角坐标
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            
            # 获取类别颜色
            color = COLOR_PALETTE[int(label) % len(COLOR_PALETTE)]
            color = [c/255 for c in color]  # 转换为matplotlib需要的0-1范围
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x_min, y_min), box_width, box_height, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax1.add_patch(rect)
            
            # 添加类别标签
            class_name = class_names[int(label)]
            ax1.text(
                x_min, y_min - 5, class_name, 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # 显示图像和预测结果
    ax2.imshow(img_np)
    ax2.set_title(f"Predictions - {img_id}")
    
    # 在图像上绘制预测边界框
    if predictions is not None and len(predictions) > 0:
        for i in range(len(predictions)):
            # 获取预测框和置信度
            box = predictions[i, :4].cpu().numpy()
            conf = predictions[i, 4].cpu().numpy()
            
            if conf < conf_threshold:
                continue
            
            # 获取类别分数和预测的类别
            cls_scores = predictions[i, 5:].cpu().numpy()
            cls_id = np.argmax(cls_scores)
            
            # 将YOLO格式的框转换为矩形框
            x1, y1, x2, y2 = box
            
            # 获取类别颜色
            color = COLOR_PALETTE[int(cls_id) % len(COLOR_PALETTE)]
            color = [c/255 for c in color]  # 转换为matplotlib需要的0-1范围
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax2.add_patch(rect)
            
            # 添加类别标签和置信度
            class_name = class_names[int(cls_id)]
            ax2.text(
                x1, y1 - 5, f"{class_name} {conf:.2f}", 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # 隐藏坐标轴
    ax1.axis('off')
    ax2.axis('off')
    
    # 设置紧凑布局
    plt.tight_layout()
    
    return fig

def visualize_test_predictions(image, detections, color_palette=None):
    """可视化测试图像上的检测结果"""
    if color_palette is None:
        color_palette = COLOR_PALETTE
    
    # 创建图像副本
    img_vis = image.copy()
    
    # 加载中文字体
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
    except:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
    
    # 在图像上绘制检测框
    for det in detections:
        # 获取框、类别和置信度
        box = det['bbox']  # [x1, y1, x2, y2]
        category_id = det['category_id']
        category_name = det['category_name']
        score = det['score']
        
        # 获取类别颜色
        color = color_palette[category_id % len(color_palette)]
        
        # 绘制边界框
        x1, y1, x2, y2 = box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
        
        # 添加类别标签和置信度
        label = f"{category_name} {score:.2f}"
        
        # 获取文本框大小
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制文本背景
        cv2.rectangle(
            img_vis, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # 绘制文本
        cv2.putText(
            img_vis, 
            label, 
            (x1, y1 - 5), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness, 
            cv2.LINE_AA
        )
    
    return img_vis

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


# filepath: yolo_code/data_preparation.py
import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from config import *

class VOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None, is_test=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_test = is_test
        
        # 根据image_set确定图像ID列表
        self.img_ids = []
        list_file = self.root_dir / "ImageSets" / "Main" / f"{image_set}.txt"
        
        with open(list_file, 'r') as f:
            for line in f:
                # 处理行，提取图像ID
                img_id = line.strip().split()[0]
                self.img_ids.append(img_id)
        
        # 类别名称到索引的映射
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        
        # 读取图像
        img_path = self.root_dir / "JPEGImages" / f"{img_id}.jpg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        
        if not self.is_test:
            # 读取标注
            anno_path = self.root_dir / "Annotations" / f"{img_id}.xml"
            boxes, labels = self._parse_annotation(anno_path)
            
            # 应用变换
            if self.transform:
                transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            
            # 转换为torch张量
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # 创建目标字典
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([index]),
                'orig_size': torch.as_tensor([height, width], dtype=torch.int64)
            }
            
            return img, target, img_id
        else:
            # 测试模式，只返回图像
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            
            return img, img_id, (height, width)
    
    def _parse_annotation(self, anno_path):
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:  # 跳过difficult=1的对象
                continue
                
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 归一化坐标
            img_width = float(root.find('size/width').text)
            img_height = float(root.find('size/height').text)
            
            # YOLO格式：x_center, y_center, width, height (归一化)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            boxes.append([x_center, y_center, width, height])
            labels.append(self.class_to_idx[name])
        
        return boxes, labels

def create_data_loaders():
    # 训练数据增强
    train_transform = A.Compose([
        A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.1, 1.0)),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # 验证数据增强
    val_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # 测试数据增强
    test_transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # 创建数据集
    train_dataset = VOCDataset(
        root_dir=TRAIN_DATA_PATH,
        image_set='train',
        transform=train_transform
    )
    
    val_dataset = VOCDataset(
        root_dir=TRAIN_DATA_PATH,
        image_set='val',
        transform=val_transform
    )
    
    test_dataset = VOCDataset(
        root_dir=TEST_DATA_PATH,
        image_set='test',
        transform=test_transform,
        is_test=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_test_fn
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """自定义批处理函数，处理不同大小的图像和标注"""
    images = []
    targets = []
    img_ids = []
    
    for img, target, img_id in batch:
        images.append(img)
        targets.append(target)
        img_ids.append(img_id)
    
    return images, targets, img_ids

def collate_test_fn(batch):
    """测试集的批处理函数"""
    images = []
    img_ids = []
    orig_sizes = []
    
    for img, img_id, orig_size in batch:
        images.append(img)
        img_ids.append(img_id)
        orig_sizes.append(orig_size)
    
    images = torch.stack(images, 0)
    
    return images, img_ids, orig_sizes

def analyze_dataset():
    """分析数据集，绘制类别分布等统计图表"""
    # 统计各类别的样本数量
    class_counts = {cls: 0 for cls in VOC_CLASSES}
    
    # 处理训练集
    anno_dir = TRAIN_DATA_PATH / "Annotations"
    for xml_file in tqdm(list(anno_dir.glob("*.xml")), desc="Analyzing dataset"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in class_counts:
                class_counts[name] += 1
    
    # 绘制类别分布条形图
    plt.figure(figsize=(14, 7))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('类别', fontproperties='SimHei', fontsize=12)
    plt.ylabel('样本数量', fontproperties='SimHei', fontsize=12)
    plt.title('VOC2007训练集中各类别样本分布', fontproperties='SimHei', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(VISUALIZATION_PATH / "class_distribution.png"))
    
    # 分析边界框大小分布
    box_areas = []
    box_aspects = []
    
    for xml_file in tqdm(list(anno_dir.glob("*.xml")), desc="Analyzing bounding boxes"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        width = float(root.find('size/width').text)
        height = float(root.find('size/height').text)
        
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 计算相对面积和宽高比
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            area = box_width * box_height
            aspect = box_width / box_height if box_height > 0 else 0
            
            box_areas.append(area)
            box_aspects.append(aspect)
    
    # 绘制边界框面积直方图
    plt.figure(figsize=(10, 6))
    plt.hist(box_areas, bins=50, alpha=0.75, color='blue')
    plt.xlabel('相对边界框面积', fontproperties='SimHei', fontsize=12)
    plt.ylabel('数量', fontproperties='SimHei', fontsize=12)
    plt.title('边界框相对面积分布', fontproperties='SimHei', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(VISUALIZATION_PATH / "bbox_area_distribution.png"))
    
    # 绘制边界框宽高比直方图
    plt.figure(figsize=(10, 6))
    plt.hist(box_aspects, bins=50, alpha=0.75, color='green')
    plt.xlabel('边界框宽高比', fontproperties='SimHei', fontsize=12)
    plt.ylabel('数量', fontproperties='SimHei', fontsize=12)
    plt.title('边界框宽高比分布', fontproperties='SimHei', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(VISUALIZATION_PATH / "bbox_aspect_distribution.png"))
    
    print(f"分析完成，图表已保存到 {VISUALIZATION_PATH}")

if __name__ == "__main__":
    # 分析数据集并生成可视化图表
    analyze_dataset()
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = create_data_loaders()
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")


    import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, to_2tuple
import math
from config import *

class CBAM(nn.Module):
    """CBAM注意力机制"""
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()
        
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid_channel(avg_out + max_out)
        
        x = x * channel_out
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv_spatial(spatial_out)
        spatial_out = self.sigmoid_spatial(spatial_out)
        
        x = x * spatial_out
        
        return x

class ConvBNSiLU(nn.Module):
    """标准卷积块，包含Conv+BN+SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.conv1 = ConvBNSiLU(in_channels, in_channels // 2, 1)
        self.conv2 = ConvBNSiLU(in_channels // 2, out_channels, 3)
        self.use_add = in_channels == out_channels
        
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPBlock(nn.Module):
    """CSP模块"""
    def __init__(self, in_channels, out_channels, num_resblocks=1, add_identity=True):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_channels, out_channels // 2)
        self.conv2 = ConvBNSiLU(in_channels, out_channels // 2)
        
        resblock_layers = []
        for _ in range(num_resblocks):
            resblock_layers.append(ResidualBlock(out_channels // 2))
        self.resblocks = nn.Sequential(*resblock_layers)
        
        if USE_CBAM:
            self.cbam = CBAM(out_channels)
        
        self.conv3 = ConvBNSiLU(out_channels, out_channels)
        self.add_identity = add_identity and in_channels == out_channels
    
    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.resblocks(y1)
        
        y2 = self.conv2(x)
        
        y = torch.cat([y1, y2], dim=1)
        y = self.conv3(y)
        
        if USE_CBAM:
            y = self.cbam(y)
            
        if self.add_identity:
            y = y + x
            
        return y

class SwinTransformerLayer(nn.Module):
    """Swin Transformer层"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix=None):
        B, C, H, W = x.shape
        
        # 将特征图转换为序列形式
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        
        shortcut = x
        x = self.norm1(x)
        
        # 如果是移位窗口注意力
        if self.shift_size > 0:
            # 计算注意力掩码
            if mask_matrix is None:
                mask_matrix = torch.zeros((1, H, W, 1), device=x.device)
                h_slices = (slice(0, -self.window_size),
                           slice(-self.window_size, -self.shift_size),
                           slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                           slice(-self.window_size, -self.shift_size),
                           slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        mask_matrix[:, h, w, :] = cnt
                        cnt += 1
                mask_matrix = mask_matrix.view(1, H // self.window_size, self.window_size, 
                                             W // self.window_size, self.window_size, 1)
                mask_matrix = mask_matrix.permute(0, 1, 3, 2, 4, 5).contiguous()
                mask_matrix = mask_matrix.view(-1, self.window_size * self.window_size)
                attn_mask = mask_matrix.unsqueeze(1) - mask_matrix.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            else:
                attn_mask = mask_matrix
            
            # 循环移位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            # 分割窗口
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            
            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
            
            # 合并窗口
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            
            # 反向循环移位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            # 分割窗口
            x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            
            # W-MSA
            attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C
            
            # 合并窗口
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # 转换回特征图形式
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        
        return x

class WindowAttention(nn.Module):
    """窗口多头自注意力模块"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # 获取窗口内每个token的相对坐标
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # 计算相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 平移到从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, self.num_heads, N, C // self.num_heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置编码
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 添加注意力掩码（对于移位窗口）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP模块"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """将特征图分割成不重叠的窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """将窗口还原为特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer块，包含一个W-MSA和一个SW-MSA"""
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        
        self.window_size = window_size
        self.attn_layers = nn.ModuleList([
            SwinTransformerLayer(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path),
            SwinTransformerLayer(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path),
        ])
    
    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        return x

class YOLOv7SwinBackbone(nn.Module):
    """使用Swin Transformer作为骨干网络的YOLOv7"""
    def __init__(self):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = ConvBNSiLU(3, 32, 3, 1)
        self.conv2 = ConvBNSiLU(32, 64, 3, 2)  # 下采样
        self.conv3 = ConvBNSiLU(64, 64, 3, 1)
        
        # 第一个阶段
        self.stage1_csp = CSPBlock(64, 128, num_resblocks=2)
        self.stage1_down = ConvBNSiLU(128, 128, 3, 2)  # 下采样
        self.stage1_swin = SwinTransformerBlock(128, num_heads=4, window_size=7)
        
        # 第二个阶段
        self.stage2_csp = CSPBlock(128, 256, num_resblocks=4)
        self.stage2_down = ConvBNSiLU(256, 256, 3, 2)  # 下采样
        self.stage2_swin = SwinTransformerBlock(256, num_heads=8, window_size=7)
        
        # 第三个阶段
        self.stage3_csp = CSPBlock(256, 512, num_resblocks=8)
        self.stage3_down = ConvBNSiLU(512, 512, 3, 2)  # 下采样
        self.stage3_swin = SwinTransformerBlock(512, num_heads=16, window_size=7)
        
        # 第四个阶段
        self.stage4_csp = CSPBlock(512, 1024, num_resblocks=4)
        
    def forward(self, x):
        # 初始卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 第一个阶段
        x = self.stage1_csp(x)
        x1 = x  # 保存特征图用于FPN
        x = self.stage1_down(x)
        x = self.stage1_swin(x)
        
        # 第二个阶段
        x = self.stage2_csp(x)
        x2 = x  # 保存特征图用于FPN
        x = self.stage2_down(x)
        x = self.stage2_swin(x)
        
        # 第三个阶段
        x = self.stage3_csp(x)
        x3 = x  # 保存特征图用于FPN
        x = self.stage3_down(x)
        x = self.stage3_swin(x)
        
        # 第四个阶段
        x = self.stage4_csp(x)
        x4 = x
        
        return x1, x2, x3, x4

class YOLOv7SwinNeck(nn.Module):
    """YOLOv7的特征金字塔网络（FPN）和路径聚合网络（PAN）结合"""
    def __init__(self):
        super().__init__()
        
        # FPN上采样路径
        self.up_conv1 = ConvBNSiLU(1024, 512, 1)
        self.up_conv2 = ConvBNSiLU(1024, 512, 1)
        self.up_conv3 = ConvBNSiLU(512, 256, 1)
        
        # FPN融合层
        self.up_fusion1 = CSPBlock(1024, 512, num_resblocks=2)
        self.up_fusion2 = CSPBlock(512, 256, num_resblocks=2)
        self.up_fusion3 = CSPBlock(256, 128, num_resblocks=2)
        
        # PAN下采样路径
        self.down_conv1 = ConvBNSiLU(256, 256, 3, 2)
        self.down_conv2 = ConvBNSiLU(512, 512, 3, 2)
        
        # PAN融合层
        self.down_fusion1 = CSPBlock(512, 512, num_resblocks=2)
        self.down_fusion2 = CSPBlock(1024, 1024, num_resblocks=2)
        
    def forward(self, features):
        x1, x2, x3, x4 = features
        
        # FPN: 自上而下路径
        p4 = self.up_conv1(x4)
        p4_up = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = torch.cat([p4_up, x3], dim=1)
        p3 = self.up_fusion1(p3)
        
        p3_up = self.up_conv2(p3)
        p3_up = F.interpolate(p3_up, scale_factor=2, mode='nearest')
        p2 = torch.cat([p3_up, x2], dim=1)
        p2 = self.up_fusion2(p2)
        
        p2_up = self.up_conv3(p2)
        p2_up = F.interpolate(p2_up, scale_factor=2, mode='nearest')
        p1 = torch.cat([p2_up, x1], dim=1)
        p1 = self.up_fusion3(p1)
        
        # PAN: 自下而上路径
        p1_down = self.down_conv1(p1)
        p2 = torch.cat([p1_down, p2], dim=1)
        p2 = self.down_fusion1(p2)
        
        p2_down = self.down_conv2(p2)
        p3 = torch.cat([p2_down, p3], dim=1)
        p3 = self.down_fusion2(p3)
        
        return p1, p2, p3

class ScalePrediction(nn.Module):
    """尺度预测层"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # 每个预测框有5+num_classes个参数：x, y, w, h, obj_conf, class_probs
        self.pred = nn.Conv2d(in_channels, 3 * (5 + num_classes), kernel_size=1)
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.pred(x)

class YOLOv7SwinHead(nn.Module):
    """YOLOv7的检测头"""
    def __init__(self, num_classes):
        super().__init__()
        
        # 检测头卷积层
        self.head_conv1 = ConvBNSiLU(128, 256, 3, 1)
        self.head_conv2 = ConvBNSiLU(512, 512, 3, 1)
        self.head_conv3 = ConvBNSiLU(1024, 1024, 3, 1)
        
        # 尺度预测层
        self.scale_pred1 = ScalePrediction(256, num_classes)
        self.scale_pred2 = ScalePrediction(512, num_classes)
        self.scale_pred3 = ScalePrediction(1024, num_classes)
        
    def forward(self, features):
        p1, p2, p3 = features
        
        # 对每个特征图进行预测
        p1 = self.head_conv1(p1)
        p2 = self.head_conv2(p2)
        p3 = self.head_conv3(p3)
        
        pred1 = self.scale_pred1(p1)
        pred2 = self.scale_pred2(p2)
        pred3 = self.scale_pred3(p3)
        
        return [pred1, pred2, pred3]

class YOLOv7Swin(nn.Module):
    """完整的YOLOv7-Swin模型"""
    def __init__(self, num_classes=20):
        super().__init__()
        
        self.backbone = YOLOv7SwinBackbone()
        self.neck = YOLOv7SwinNeck()
        self.head = YOLOv7SwinHead(num_classes)
        
        # 锚点，三个尺度，每个尺度三个锚点
        self.anchors = torch.tensor([
            # 小尺度
            [[10, 13], [16, 30], [33, 23]],
            # 中尺度
            [[30, 61], [62, 45], [59, 119]],
            # 大尺度
            [[116, 90], [156, 198], [373, 326]],
        ]).float()
        
        self.strides = torch.tensor([8, 16, 32])
        self.num_classes = num_classes
        
    def forward(self, x):
        # 获取批次大小和特征图尺寸
        batch_size = x.shape[0]
        
        # 通过主干网络
        backbone_features = self.backbone(x)
        
        # 通过颈部网络
        neck_features = self.neck(backbone_features)
        
        # 通过检测头
        predictions = self.head(neck_features)
        
        # 如果是训练模式，直接返回原始预测结果
        if self.training:
            return predictions
        
        # 如果是推理模式，处理预测结果
        device = x.device
        grids = []
        strides = []
        
        # 转换预测结果的格式并解码
        for i, pred in enumerate(predictions):
            # 获取特征图尺寸
            _, _, h, w = pred.shape
            
            # 尺度和锚点
            stride = self.strides[i]
            anchors = self.anchors[i] / stride
            
            # 重塑预测结果
            pred = pred.view(batch_size, 3, 5 + self.num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
            
            # 创建网格
            grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
            grid = torch.stack((grid_x, grid_y), 2).view(1, 1, h, w, 2).float().to(device)
            grids.append(grid)
            strides.append(torch.full((1, 1, h, w, 1), stride).to(device))
            
            # 解码预测结果
            pred[..., 0:2] = (torch.sigmoid(pred[..., 0:2]) + grid) * stride
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * anchors.view(1, 3, 1, 1, 2).to(device) * stride
            pred[..., 4:] = torch.sigmoid(pred[..., 4:])
            
            predictions[i] = pred.view(batch_size, -1, 5 + self.num_classes)
        
        # 合并所有尺度的预测结果
        output = torch.cat(predictions, dim=1)
        
        return output

def build_model():
    """构建目标检测模型"""
    model = YOLOv7Swin(num_classes=len(VOC_CLASSES))
    return model

if __name__ == "__main__":
    # 测试模型
    model = build_model()
    x = torch.randn(2, 3, 640, 640)
    output = model(x)
    print(f"输入形状: {x.shape}")
    
    if isinstance(output, list):
        print("训练模式:")
        for i, out in enumerate(output):
            print(f"  输出 {i+1} 形状: {out.shape}")
    else:
        print(f"推理模式, 输出形状: {output.shape}")


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


    import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import logging
from tqdm import tqdm
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import cv2

from config import *
from model import build_model
from data_preparation import create_data_loaders
from utils.metrics import compute_ap, box_iou, ap_per_class, xywh2xyxy
from utils.visualization import visualize_predictions

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname=FONT_PATH)

# 配置日志
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH / "validation.log"),
        logging.StreamHandler()
    ]
)

class Validator:
    def __init__(self, model, val_loader, device, confidence_threshold=0.25, iou_threshold=0.45):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 确保结果目录存在
        self.validation_results_path = METRIC_PATH / "validation"
        self.validation_images_path = VISUALIZATION_PATH / "validation"
        self.validation_results_path.mkdir(parents=True, exist_ok=True)
        self.validation_images_path.mkdir(parents=True, exist_ok=True)
        
        # 存储结果
        self.class_ap = {cls_name: 0.0 for cls_name in VOC_CLASSES}
        self.predictions = []
        self.targets = []
        self.confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
    def validate(self):
        logging.info("开始验证...")
        self.model.eval()
        
        # 收集所有类别的预测和真实标签
        all_pred_boxes = []  # 所有预测框
        all_pred_labels = []  # 所有预测标签
        all_pred_scores = []  # 所有预测分数
        all_true_boxes = []  # 所有真实框
        all_true_labels = []  # 所有真实标签
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            
            for batch_idx, (images, targets, img_ids) in progress_bar:
                # 将图像和目标移到设备
                images = [img.to(self.device) for img in images]
                for target in targets:
                    for k, v in target.items():
                        target[k] = v.to(self.device)
                
                # 前向传播获取预测结果
                predictions = self.model(torch.stack(images))
                
                # 处理每张图像的预测和目标
                for i, (pred, target, img_id) in enumerate(zip(predictions, targets, img_ids)):
                    # 筛选满足置信度阈值的检测结果
                    mask = pred[:, 4] > self.confidence_threshold
                    pred_filtered = pred[mask]
                    
                    if len(pred_filtered) > 0:
                        # 获取预测框和类别
                        pred_boxes = pred_filtered[:, :4]  # x, y, w, h 格式
                        pred_scores = pred_filtered[:, 4]
                        pred_class_scores = pred_filtered[:, 5:]
                        pred_labels = torch.argmax(pred_class_scores, dim=1)
                        
                        # 转换为 xyxy 格式
                        pred_boxes_xyxy = xywh2xyxy(pred_boxes)
                        
                        # 保存预测结果
                        for box, label, score in zip(pred_boxes_xyxy.cpu().numpy(), 
                                                    pred_labels.cpu().numpy(), 
                                                    pred_scores.cpu().numpy()):
                            all_pred_boxes.append(box)
                            all_pred_labels.append(label)
                            all_pred_scores.append(score)
                    
                    # 获取真实标注
                    true_boxes = target['boxes']  # yolo 格式 (x_center, y_center, width, height)
                    true_labels = target['labels']
                    
                    # 转换为 xyxy 格式
                    true_boxes_xyxy = xywh2xyxy(true_boxes)
                    
                    for box, label in zip(true_boxes_xyxy.cpu().numpy(), true_labels.cpu().numpy()):
                        all_true_boxes.append(box)
                        all_true_labels.append(label)
                    
                    # 可视化每50批次的第一个样本
                    if batch_idx % 50 == 0 and i == 0:
                        sample_img = images[i].detach().cpu()
                        sample_pred = pred_filtered.detach().cpu() if len(pred_filtered) > 0 else None
                        sample_target = {k: v.detach().cpu() for k, v in target.items()}
                        
                        # 可视化预测结果
                        fig = visualize_predictions(
                            sample_img, 
                            sample_pred, 
                            sample_target, 
                            img_id,
                            VOC_CLASSES
                        )
                        
                        save_path = self.validation_images_path / f"val_sample_{batch_idx}.png"
                        plt.savefig(str(save_path))
                        plt.close(fig)
        
        # 转换为numpy数组
        all_pred_boxes = np.array(all_pred_boxes)
        all_pred_labels = np.array(all_pred_labels)
        all_pred_scores = np.array(all_pred_scores)
        all_true_boxes = np.array(all_true_boxes)
        all_true_labels = np.array(all_true_labels)
        
        # 计算各类别AP和mAP
        stats = []
        
        # 检查是否有预测结果
        if len(all_pred_boxes) > 0:
            # 计算各类别的AP
            ap, p, r = ap_per_class(
                tp=all_pred_boxes,
                conf=all_pred_scores,
                pred_cls=all_pred_labels,
                target_cls=all_true_labels,
                target_boxes=all_true_boxes,
                iou_thres=self.iou_threshold
            )
            
            # 存储各类别AP
            for i, class_name in enumerate(VOC_CLASSES):
                self.class_ap[class_name] = ap[i] if i < len(ap) else 0.0
            
            # 计算mAP
            map_score = ap.mean()
            
            # 打印结果
            logging.info(f"验证完成，mAP@{self.iou_threshold}: {map_score:.4f}")
            
            # 可视化各类别的AP
            self.visualize_class_ap()
            
            # 可视化PR曲线
            self.visualize_pr_curves(p, r)
            
            # 可视化混淆矩阵
            self.calculate_confusion_matrix(all_pred_labels, all_true_labels)
            self.visualize_confusion_matrix()
            
            # 保存验证结果
            self.save_validation_results(map_score)
        else:
            logging.warning("没有检测到任何目标，无法计算AP")
            map_score = 0.0
        
        return map_score
    
    def visualize_class_ap(self):
        """可视化各类别的AP"""
        plt.figure(figsize=(14, 8))
        classes = list(self.class_ap.keys())
        ap_values = list(self.class_ap.values())
        
        # 创建条形图
        bars = plt.bar(classes, ap_values, color='skyblue')
        
        # 在条形上方显示具体数值
        for bar, ap in zip(bars, ap_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{ap:.3f}', ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.xlabel('类别', fontproperties=font, fontsize=12)
        plt.ylabel('AP', fontproperties=font, fontsize=12)
        plt.title('各类别的平均精度(AP)', fontproperties=font, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.validation_results_path / "class_ap.png"
        plt.savefig(str(save_path))
        plt.close()
    
    def visualize_pr_curves(self, p, r):
        """可视化PR曲线"""
        plt.figure(figsize=(12, 10))
        
        # 绘制每个类别的PR曲线
        for i, class_name in enumerate(VOC_CLASSES):
            if i < len(p) and i < len(r):
                plt.plot(r[i], p[i], linewidth=2, label=f"{class_name} (AP={self.class_ap[class_name]:.3f})")
        
        plt.xlabel('召回率', fontproperties=font, fontsize=12)
        plt.ylabel('精确率', fontproperties=font, fontsize=12)
        plt.title('各类别的精确率-召回率曲线', fontproperties=font, fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(linestyle='--', alpha=0.6)
        plt.legend(loc='lower left', prop=font)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.validation_results_path / "pr_curves.png"
        plt.savefig(str(save_path))
        plt.close()
        
        # 单独绘制每个类别的PR曲线，便于观察
        os.makedirs(str(self.validation_results_path / "pr_curves"), exist_ok=True)
        
        for i, class_name in enumerate(VOC_CLASSES):
            if i < len(p) and i < len(r):
                plt.figure(figsize=(8, 6))
                plt.plot(r[i], p[i], linewidth=2, color='blue')
                plt.xlabel('召回率', fontproperties=font, fontsize=12)
                plt.ylabel('精确率', fontproperties=font, fontsize=12)
                plt.title(f'{class_name} 精确率-召回率曲线 (AP={self.class_ap[class_name]:.3f})', fontproperties=font, fontsize=14)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.grid(linestyle='--', alpha=0.6)
                plt.tight_layout()
                
                # 保存图像
                save_path = self.validation_results_path / "pr_curves" / f"{class_name}_pr_curve.png"
                plt.savefig(str(save_path))
                plt.close()
    
    def calculate_confusion_matrix(self, pred_labels, true_labels):
        """计算混淆矩阵"""
        self.confusion_matrix = confusion_matrix(
            true_labels, 
            pred_labels, 
            labels=list(range(NUM_CLASSES))
        )
    
    def visualize_confusion_matrix(self):
        """可视化混淆矩阵"""
        plt.figure(figsize=(16, 14))
        
        # 计算归一化的混淆矩阵
        cm_normalized = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # 使用seaborn绘制热图
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=VOC_CLASSES,
            yticklabels=VOC_CLASSES
        )
        
        plt.xlabel('预测类别', fontproperties=font, fontsize=12)
        plt.ylabel('真实类别', fontproperties=font, fontsize=12)
        plt.title('混淆矩阵', fontproperties=font, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.validation_results_path / "confusion_matrix.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_validation_results(self, map_score):
        """保存验证结果到JSON文件"""
        results = {
            'mAP': float(map_score),
            'class_AP': {k: float(v) for k, v in self.class_ap.items()},
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold
        }
        
        with open(str(self.validation_results_path / "validation_results.json"), 'w') as f:
            json.dump(results, f, indent=4)

def main():
    # 创建数据加载器
    _, val_loader, _ = create_data_loaders()
    
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
        logging.warning(f"找不到模型权重文件: {best_model_path}，使用随机初始化的模型进行验证")
    
    # 创建验证器
    validator = Validator(
        model=model,
        val_loader=val_loader,
        device=DEVICE,
        confidence_threshold=CONF_THRESH,
        iou_threshold=IOU_THRESH
    )
    
    # 执行验证
    map_score = validator.validate()
    
    logging.info(f"验证完成，mAP@{IOU_THRESH}: {map_score:.4f}")

if __name__ == "__main__":
    main()