# --------------------- 
# # 黑电平校正模块（Black Level Correction） 
# # ✅ 通常应在 demosaic 之前做，用于去除传感器底噪。
"""import numpy as np
from utils.image_io import save_image_debug

def apply(raw, config, debug_path=None):
    black_level = config.get("black_level",64)
    
    if debug_path:         
         save_image_debug(raw, debug_path.replace(".png", "_before.png"), scale=True)
    corrected = raw - black_level
    corrected = np.clip(corrected, 0, None)
    if debug_path:         
        save_image_debug(corrected, debug_path.replace(".png", "_after.png"), scale=True)
    
    return corrected
    """

import numpy as np

def apply(raw, config):
    black_level = config.get("black_level", 64)
    
    # 获取位深配置
    bit_depth_cfg = config.get('bit_depth_management', {})
    input_bits = bit_depth_cfg.get('sensor_native', 10)
    processing_bits = bit_depth_cfg.get('raw_processing', 16)
    
    # BLC处理：先减去黑电平，再扩展位深
    corrected = raw.astype(np.float32) - black_level
    corrected = np.maximum(corrected, 0)
    
    # 位深扩展到16bit处理精度
    if processing_bits > input_bits:
        scale_factor = (2**processing_bits - 1) / (2**input_bits - 1)
        corrected = corrected * scale_factor
        print(f"BLC: {input_bits}bit → {processing_bits}bit, 缩放={scale_factor:.2f}")
    
    print(f"BLC: 输出范围 [{corrected.min():.1f}, {corrected.max():.1f}]")
    return corrected.astype(np.float32)
