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
    
    # BLC处理
    corrected = raw.astype(np.float32) - black_level
    corrected = np.maximum(corrected, 0)
    
    # 位深扩展：10bit → 12bit
    bit_depth_cfg = config.get('bit_depth_management', {})
    output_bit_depth = bit_depth_cfg.get('blc_output', 12)
    
    if output_bit_depth > 10:
        scale_factor = (2**output_bit_depth - 1) / (2**10 - 1)  # 4095/1023 ≈ 4.0
        corrected = corrected * scale_factor
        print(f"BLC: 位深扩展 10bit → {output_bit_depth}bit, 缩放因子: {scale_factor:.2f}")
        print(f"BLC: 输出范围 [{corrected.min():.1f}, {corrected.max():.1f}]")
    
    return corrected.astype(np.float32)
