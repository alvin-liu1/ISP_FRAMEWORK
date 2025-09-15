import numpy as np

def apply(rgb, config):
    print(f"WB: 输入Float32 HDR范围 [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    method = config.get("method", "manual")
    
    if method == "manual":
        gains = config.get("gains", [1.0, 1.0, 1.0])
        rgb[:, :, 0] *= gains[0]  # R
        rgb[:, :, 1] *= gains[1]  # G  
        rgb[:, :, 2] *= gains[2]  # B
        print(f"WB: 应用手动增益 R={gains[0]:.2f}, G={gains[1]:.2f}, B={gains[2]:.2f}")
        
    elif method == "gray_world":
        # Gray World算法实现
        min_threshold = config.get("wb_min_luminance_threshold", 0.05)
        max_threshold = config.get("wb_max_luminance_threshold", 0.99)
        
        # 计算有效像素掩膜（避免过暗和过亮区域）
        luminance = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        valid_mask = (luminance > min_threshold) & (luminance < max_threshold)
        
        if np.sum(valid_mask) > 100:  # 确保有足够的有效像素
            # 计算各通道平均值
            r_mean = np.mean(rgb[:,:,0][valid_mask])
            g_mean = np.mean(rgb[:,:,1][valid_mask])
            b_mean = np.mean(rgb[:,:,2][valid_mask])
            
            # 以绿色通道为基准计算增益
            r_gain = g_mean / r_mean if r_mean > 0 else 1.0
            g_gain = 1.0
            b_gain = g_mean / b_mean if b_mean > 0 else 1.0
            
            # 限制增益范围，避免过度校正
            max_gain = config.get("max_gain", 3.0)
            min_gain = config.get("min_gain", 0.3)
            
            r_gain = np.clip(r_gain, min_gain, max_gain)
            b_gain = np.clip(b_gain, min_gain, max_gain)
            
            # 应用增益
            rgb[:, :, 0] *= r_gain
            rgb[:, :, 1] *= g_gain  
            rgb[:, :, 2] *= b_gain
            
            print(f"WB: Gray World增益 R={r_gain:.2f}, G={g_gain:.2f}, B={b_gain:.2f}")
            print(f"WB: 有效像素数量: {np.sum(valid_mask)}")
        else:
            print("WB: Gray World失败，有效像素不足，跳过处理")
    
    elif method == "white_patch":
        # White Patch算法实现
        percentile = config.get("white_patch_percentile", 99.5)
        
        # 找到各通道的高亮区域
        r_max = np.percentile(rgb[:,:,0], percentile)
        g_max = np.percentile(rgb[:,:,1], percentile)
        b_max = np.percentile(rgb[:,:,2], percentile)
        
        # 以最亮的通道为基准
        max_channel = max(r_max, g_max, b_max)
        
        if max_channel > 0:
            r_gain = max_channel / r_max if r_max > 0 else 1.0
            g_gain = max_channel / g_max if g_max > 0 else 1.0
            b_gain = max_channel / b_max if b_max > 0 else 1.0
            
            # 限制增益范围
            max_gain = config.get("max_gain", 3.0)
            min_gain = config.get("min_gain", 0.3)
            
            r_gain = np.clip(r_gain, min_gain, max_gain)
            g_gain = np.clip(g_gain, min_gain, max_gain)
            b_gain = np.clip(b_gain, min_gain, max_gain)
            
            # 应用增益
            rgb[:, :, 0] *= r_gain
            rgb[:, :, 1] *= g_gain  
            rgb[:, :, 2] *= b_gain
            
            print(f"WB: White Patch增益 R={r_gain:.2f}, G={g_gain:.2f}, B={b_gain:.2f}")
        else:
            print("WB: White Patch失败，图像过暗，跳过处理")
    
    print(f"WB: 输出Float32 HDR范围 [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    return rgb.astype(np.float32)
