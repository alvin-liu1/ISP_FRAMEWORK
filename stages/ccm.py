# ---------------------
# 颜色校正矩阵模块（Color Correction Matrix）
# ✅ 一般在白平衡之后执行，将图像从 sensor RGB 映射到 sRGB。

import numpy as np

def apply(rgb, config):
    # 确保输入 rgb 是 float32 类型，并裁剪到 0-1 范围 (来自 WB)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    
    matrix = np.array(config["matrix"], dtype=np.float32)
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    
    corrected = np.dot(flat, matrix.T)
    
    
    # 关键修正：CCM 运算后裁剪回 0-1 范围，并保持 float32 类型
    return np.clip(corrected.reshape(h, w, 3), 0, 1).astype(np.float32)