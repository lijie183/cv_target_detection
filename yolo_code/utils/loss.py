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