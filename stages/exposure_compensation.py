import numpy as np

def apply(raw, config):
    """软件曝光补偿"""
    if not config.get('enable', False):
        return raw
    
    mode = config.get('mode', 'auto')
    
    if mode == 'auto':
        # 分析图像亮度分布
        target_percentile = config.get('target_percentile', 85)
        target_brightness = config.get('target_brightness', 0.6)
        
        # 获取位深信息
        bit_depth_cfg = config.get('bit_depth_management', {})
        processing_bits = bit_depth_cfg.get('raw_processing', 16)
        max_value = (2**processing_bits) - 1
        
        # 计算当前亮度
        current_brightness = np.percentile(raw, target_percentile) / max_value
        
        if current_brightness > 0.01:
            gain = target_brightness / current_brightness
            gain = np.clip(gain, 0.1, 4.0)  # 限制增益范围
        else:
            gain = 1.0
            
        print(f"曝光补偿: 当前亮度={current_brightness:.3f}, 增益={gain:.2f}x")
        
    else:  # manual
        gain = config.get('manual_gain', 1.0)
    
    # 应用增益
    compensated = raw * gain
    
    # 高光保护
    highlight_threshold = config.get('highlight_threshold', 0.95)
    max_value = (2**bit_depth_cfg.get('raw_processing', 16)) - 1
    
    # 软压缩过曝区域
    over_exposed = compensated > (highlight_threshold * max_value)
    if np.any(over_exposed):
        compression = config.get('highlight_compression', 0.7)
        threshold_val = highlight_threshold * max_value
        
        excess = compensated[over_exposed] - threshold_val
        compensated[over_exposed] = threshold_val + excess * compression
        
        print(f"曝光补偿: 高光保护 {np.sum(over_exposed)} 像素")
    
    return compensated.astype(np.float32)