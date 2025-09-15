import numpy as np
from scipy.ndimage import median_filter

def apply(raw, config):
    """坏点校正 - Bayer感知版本"""
    if not config.get('enable', False):
        return raw
    
    method = config.get('method', 'median')
    threshold = config.get('threshold', 3.0)
    bayer_pattern = config.get('bayer_pattern', 'rggb').lower()
    
    if method == 'median':
        corrected = bayer_aware_dpc(raw, threshold, bayer_pattern)
        return corrected
    
    return raw

def bayer_aware_dpc(raw, threshold, pattern):
    """Bayer感知的坏点校正"""
    h, w = raw.shape
    corrected = raw.copy()
    
    # 创建Bayer masks
    r_mask, g_mask, b_mask = create_bayer_masks(h, w, pattern)
    
    # 分别处理每个颜色通道
    bad_count = 0
    
    # 处理R通道
    bad_count += fix_channel_bad_pixels(corrected, r_mask, threshold)
    
    # 处理G通道  
    bad_count += fix_channel_bad_pixels(corrected, g_mask, threshold)
    
    # 处理B通道
    bad_count += fix_channel_bad_pixels(corrected, b_mask, threshold)
    
    print(f"DPC: 检测并修复 {bad_count} 个坏点 ({bad_count/raw.size*100:.3f}%)")
    
    return corrected

def fix_channel_bad_pixels(raw, channel_mask, threshold):
    """修复单个颜色通道的坏点 - 只处理暗坏点"""
    # 提取该通道的像素
    channel_pixels = raw[channel_mask]
    
    # 计算统计量
    q25, q50 = np.percentile(channel_pixels, [25, 50])
    mad = np.median(np.abs(channel_pixels - q50))
    
    # 🔥 只检测异常暗的像素（暗坏点）
    # 不处理亮坏点，避免误伤高光
    dark_threshold = q25 - threshold * mad
    bad_pixels = channel_pixels < dark_threshold
    
    if np.any(bad_pixels):
        # 用中值替换暗坏点
        channel_pixels[bad_pixels] = q50
        raw[channel_mask] = channel_pixels
    
    return np.sum(bad_pixels)

def create_bayer_masks(h, w, pattern):
    """创建Bayer模式mask"""
    r_mask = np.zeros((h, w), dtype=bool)
    g_mask = np.zeros((h, w), dtype=bool) 
    b_mask = np.zeros((h, w), dtype=bool)
    
    if pattern == "rggb":
        r_mask[0::2, 0::2] = True  # R
        g_mask[0::2, 1::2] = True  # G
        g_mask[1::2, 0::2] = True  # G
        b_mask[1::2, 1::2] = True  # B
    elif pattern == "bggr":
        b_mask[0::2, 0::2] = True  # B
        g_mask[0::2, 1::2] = True  # G
        g_mask[1::2, 0::2] = True  # G
        r_mask[1::2, 1::2] = True  # R
    # 其他模式...
    
    return r_mask, g_mask, b_mask



