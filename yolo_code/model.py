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