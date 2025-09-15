# ---------------------
# Gamma 校正模块
# ✅ 通常放在 ISP 流程末尾，使图像对比更符合人眼感知。

# stages/gamma.py
# ---------------------
# Gamma 校正模块
# ✅ 通常放在 ISP 流程末尾，使图像对比更符合人眼感知。

import numpy as np

def apply(rgb, config):
    gamma_value = config.get("value", 2.2)
    curve_type = config.get("curve_type", "standard")
    
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    
    if curve_type == "s_curve":
        midtone_boost = config.get("midtone_boost", 0.08)
        
        # 分段gamma值
        shadow_gamma = gamma_value * 0.9   # 暗部gamma更低
        midtone_gamma = gamma_value * 0.95 # 中调gamma稍低
        highlight_gamma = gamma_value      # 高光正常
        
        # 分段处理
        shadow_mask = rgb < 0.3
        midtone_mask = (rgb >= 0.3) & (rgb < 0.7)
        highlight_mask = rgb >= 0.7
        
        corrected = np.copy(rgb)
        
        if np.any(shadow_mask):
            corrected[shadow_mask] = np.power(rgb[shadow_mask], 1.0/shadow_gamma)
        
        if np.any(midtone_mask):
            corrected[midtone_mask] = np.power(rgb[midtone_mask], 1.0/midtone_gamma)
        
        if np.any(highlight_mask):
            corrected[highlight_mask] = np.power(rgb[highlight_mask], 1.0/highlight_gamma)
        
        print(f"Gamma: S曲线分段处理")
    else:
        # 标准Gamma
        try:
            power_exponent = 1.0 / gamma_value
            corrected = np.power(rgb, power_exponent)
        except Exception as e:
            print(f"伽马校正错误: {e}")
            corrected = rgb
    
    corrected = np.clip(corrected, 0, 1)
    return corrected # 返回 0-1 范围的浮点数图像 (非线性亮度)
