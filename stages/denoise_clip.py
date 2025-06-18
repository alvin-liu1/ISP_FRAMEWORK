import numpy as np

def apply(raw, config):
    threshold = config.get("threshold", 8.0)

    # 保留 float 精度，但低于 threshold 的值直接 clip 到 0
    raw_denoised = np.where(raw < threshold, 0, raw)
    
    return raw_denoised.astype(np.float32)
