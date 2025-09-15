import numpy as np

def apply(rgb, config):
    """阴影/高光调整 - 在HDR空间处理"""
    if not config.get('enable', False):
        return rgb
    
    shadow_amount = config.get('shadow_amount', 0.0) / 100.0    # -1 to +1
    highlight_amount = config.get('highlight_amount', 0.0) / 100.0  # -1 to +1
    
    if abs(shadow_amount) < 1e-6 and abs(highlight_amount) < 1e-6:
        return rgb
    
    # 计算亮度
    luminance = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    
    # 获取HDR范围
    bit_depth_cfg = config.get('bit_depth_management', {})
    hdr_max = bit_depth_cfg.get('hdr_range', [0.0, 8.0])[1]
    
    # 归一化亮度到0-1
    norm_luminance = np.clip(luminance / hdr_max, 0, 1)
    
    # 创建平滑的阴影和高光掩膜
    shadow_mask = np.power(1.0 - norm_luminance, 2)  # 暗部权重高
    highlight_mask = np.power(norm_luminance, 2)      # 亮部权重高
    
    # 计算调整系数
    shadow_factor = 1.0 + shadow_amount * shadow_mask
    highlight_factor = 1.0 - highlight_amount * highlight_mask
    
    # 组合调整
    total_factor = shadow_factor * highlight_factor
    
    # 应用调整
    adjusted_rgb = rgb * total_factor[:,:,np.newaxis]
    
    print(f"阴影/高光: 阴影={shadow_amount:.2f}, 高光={highlight_amount:.2f}")
    return adjusted_rgb