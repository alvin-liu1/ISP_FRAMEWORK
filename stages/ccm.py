# ---------------------
# 颜色校正矩阵模块（Color Correction Matrix）
# ✅ 一般在白平衡之后执行，将图像从 sensor RGB 映射到 sRGB。

import numpy as np

def apply(rgb, config):
    rgb = rgb.astype(np.float32)
    matrix = np.array(config["matrix"], dtype=np.float32)
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    
    print(f"CCM: 输入范围 [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    # 计算亮度，识别高光区域
    luminance = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    highlight_threshold = config.get("highlight_threshold", 0.8)
    
    # 高光区域直接跳过CCM
    highlight_mask = luminance > highlight_threshold
    
    # 应用CCM矩阵
    corrected = np.dot(flat, matrix.T)
    corrected = np.maximum(corrected, 0.0)
    
    # 高光区域保持原始颜色
    corrected[highlight_mask] = flat[highlight_mask]
    
    print(f"CCM: 跳过高光像素 {np.sum(highlight_mask)} 个")
    print(f"CCM: 输出范围 [{corrected.min():.4f}, {corrected.max():.4f}]")
    
    # 饱和度增强（排除高光区域）
    saturation_boost = config.get("saturation_boost", 1.0)
    if saturation_boost != 1.0:
        corrected_reshaped = corrected.reshape(h, w, 3)
        highlight_mask_2d = highlight_mask.reshape(h, w)
        
        gray = 0.299 * corrected_reshaped[:,:,0] + 0.587 * corrected_reshaped[:,:,1] + 0.114 * corrected_reshaped[:,:,2]
        gray = np.expand_dims(gray, axis=2)
        
        saturation_enhanced = gray + (corrected_reshaped - gray) * saturation_boost
        saturation_enhanced = np.maximum(saturation_enhanced, 0.0)
        
        # 高光区域不做饱和度增强
        corrected_reshaped = np.where(
            np.expand_dims(highlight_mask_2d, axis=2),
            corrected_reshaped,  # 保持原样
            saturation_enhanced  # 应用饱和度增强
        )
        
        corrected = corrected_reshaped.reshape(-1, 3)
    
    return corrected.reshape(h, w, 3).astype(np.float32)
