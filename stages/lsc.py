

# stages/lsc.py
import numpy as np
from scipy.ndimage import gaussian_filter

def apply(raw, config):
    print(f"LSC输入: dtype={raw.dtype}, min={raw.min()}, max={raw.max()}")
    
    # 获取位深配置
    bit_depth_cfg = config.get('bit_depth_management', {})
    processing_bits = bit_depth_cfg.get('raw_processing', 16)
    max_value = (2**processing_bits) - 1
    
    h, w = raw.shape
    model_type = config.get("model_type", "cosine_fourth")
    strength = config.get("strength", 0.3)
    
    # 在16bit精度下处理，避免截断误差
    y, x = np.indices((h, w), dtype=np.float32)
    center_y, center_x = h / 2.0, w / 2.0
    
    pixel_size = config.get("pixel_size_um", 1.4)
    focal_length = config.get("focal_length_mm", 4.0)
    
    dx = (x - center_x) * pixel_size / 1000.0
    dy = (y - center_y) * pixel_size / 1000.0
    r_mm = np.sqrt(dx**2 + dy**2)
    theta = np.arctan(r_mm / focal_length)
    
    if model_type == "cosine_fourth":
        gain_map = 1.0 / (np.cos(theta) ** 4)
    else:
        gain_map = np.ones((h, w), dtype=np.float32)
    
    # 限制最大增益，避免过度放大
    max_allowed_gain = max_value / raw.max() if raw.max() > 0 else 1.5
    max_allowed_gain = min(max_allowed_gain, 1.5)  # 硬限制1.5x
    gain_map = np.clip(gain_map, 1.0, max_allowed_gain)
    
    # 应用强度控制
    gain_map = 1.0 + (gain_map - 1.0) * strength
    
    # 应用LSC校正
    corrected = raw * gain_map
    
    # 确保不超出16bit范围
    corrected = np.clip(corrected, 0, max_value)
    
    print(f"LSC: 16bit处理完成，输出范围 [{corrected.min():.1f}, {corrected.max():.1f}]")
    return corrected.astype(np.float32)
