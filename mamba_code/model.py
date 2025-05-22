import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection import FasterRCNN
from mamba_ssm.models.mixer_seq_simple import MambaBlock

class MambaFeatureExtractor(nn.Module):
    """Mamba特征提取器: 将特征图按序列处理"""
    def __init__(self, in_channels, d_model=512, depth=2):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        
        # 初始投影
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Mamba层
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, 
                      d_state=16,
                      d_conv=4,
                      expand=2)
            for _ in range(depth)
        ])
        
        # 最终投影回原始通道深度
        self.output_proj = nn.Conv2d(d_model, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # 初始投影
        x = self.input_proj(x)
        
        # 重塑以进行序列建模: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.flatten(2).permute(0, 2, 1)
        
        # 应用Mamba块
        for block in self.mamba_blocks:
            x_seq = block(x_seq)
        
        # 重塑回原始形状: (B, H*W, C) -> (B, C, H, W)
        x = x_seq.permute(0, 2, 1).view(batch, self.d_model, height, width)
        
        # 最终投影
        x = self.output_proj(x)
        
        return x

class MambaBackbone(nn.Module):
    """带Mamba增强的ResNet骨干网络"""
    def __init__(self, backbone_name='resnet50', pretrained=True, trainable_layers=3):
        super().__init__()
        
        # 获取CNN骨干网络
        if backbone_name == 'resnet50':
            # 使用预训练的ResNet50作为基础CNN特征提取器
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            # 提取各阶段
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1  # 256通道
            self.layer2 = backbone.layer2  # 512通道
            self.layer3 = backbone.layer3  # 1024通道
            self.layer4 = backbone.layer4  # 2048通道
            
            # 特征通道数
            self.out_channels = 2048
            
            # 将Mamba增强应用于选定层
            self.mamba_layer3 = MambaFeatureExtractor(in_channels=1024, d_model=512, depth=2)
            self.mamba_layer4 = MambaFeatureExtractor(in_channels=2048, d_model=512, depth=2)
            
            # 冻结一些层
            self._freeze_layers(trainable_layers)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
            
    def _freeze_layers(self, trainable_layers):
        """冻结选定层以进行迁移学习"""
        # 冻结特定层
        layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        
        # 首先冻结所有层
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
                
        # 解冻指定数量的可训练层
        for layer in layers[-trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # 对layer3应用Mamba
        x = self.layer3(x)
        x = x + self.mamba_layer3(x)  # 添加残差连接
        
        # 对layer4应用Mamba
        x = self.layer4(x)
        x = x + self.mamba_layer4(x)  # 添加残差连接
        
        return x

class MambaRCNN(nn.Module):
    """带Mamba增强骨干网络的Faster R-CNN"""
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True):
        super().__init__()
        
        # 初始化骨干网络
        self.backbone = MambaBackbone(backbone_name=backbone_name, pretrained=pretrained)
        
        # 创建锚点生成器
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # 创建ROI池化层
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # 创建Faster R-CNN模型
        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=600,
            max_size=1000,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

# 创建模型的工具函数
def create_model(num_classes=21, pretrained=True):  # 20个类 + 背景
    model = MambaRCNN(num_classes=num_classes, pretrained=pretrained)
    return model

# 测试代码
if __name__ == "__main__":
    # 测试模型
    model = create_model()
    x = torch.randn(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print("模型输出键:", output[0].keys())