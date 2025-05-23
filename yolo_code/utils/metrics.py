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