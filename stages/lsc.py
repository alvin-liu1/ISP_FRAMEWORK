

# stages/lsc.py
import numpy as np
from scipy.ndimage import gaussian_filter

def apply(raw, config):
    print(f"LSC输入: dtype={raw.dtype}, min={raw.min()}, max={raw.max()}")
    
    h, w = raw.shape
    model_type = config.get("model_type", "cosine_fourth")
    strength = config.get("strength", 0.3)
    
    # 获取位深配置
    bit_depth_cfg = config.get('bit_depth_management', {})
    input_bit_depth = bit_depth_cfg.get('blc_output', 12)  # 来自BLC的12bit
    output_bit_depth = bit_depth_cfg.get('lsc_output', 12)  # 保持12bit
    max_value = (2**input_bit_depth) - 1  # 4095 for 12-bit
    
    # 计算增益图
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
    corrected_raw = raw * gain_map
    
    # 保持在12bit范围内
    corrected_raw = np.clip(corrected_raw, 0, max_value)
    
    print(f"LSC: 保持{output_bit_depth}bit, 最大值: {max_value}")
    print(f"LSC: 输出范围 [{corrected_raw.min():.1f}, {corrected_raw.max():.1f}]")
    
    return corrected_raw.astype(np.float32)
