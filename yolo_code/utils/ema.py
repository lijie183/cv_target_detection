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