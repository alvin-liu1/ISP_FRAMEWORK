# ---------------------
# 去马赛克（Demosaic）模块
# ✅ 通常在 BLC 之后、WB/CCM 之前执行。

import cv2
import numpy as np
from scipy import ndimage

def apply(raw, config):
    method = config.get("method", "opencv_vng")
    pattern = config.get("bayer_pattern", "bggr").lower()
    
    print(f"DEBUG: 使用Demosaic算法: {method}")
    
    if method == "selective_anti_moire":
        rgb = selective_anti_moire_demosaic(raw, config)
    elif method == "opencv_ea":
        rgb = opencv_demosaic_ea(raw, config)
    else:
        rgb = opencv_demosaic(raw, config)
    
    rgb_normalized = rgb / rgb.max()
    return rgb_normalized.astype(np.float32)

def adaptive_gradient_demosaic(raw, config):
    """自适应梯度demosaic - 更好的边缘保持"""
    pattern = config.get("bayer_pattern", "rggb").lower()
    h, w = raw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # 创建Bayer masks
    r_mask, g_mask, b_mask = create_bayer_masks(h, w, pattern)
    
    # 直接复制已知像素
    rgb[:,:,0][r_mask] = raw[r_mask]  # R
    rgb[:,:,1][g_mask] = raw[g_mask]  # G  
    rgb[:,:,2][b_mask] = raw[b_mask]  # B
    
    # 插值缺失像素
    rgb = interpolate_missing_pixels(rgb, raw, r_mask, g_mask, b_mask)
    
    print("DEBUG: 使用自适应梯度demosaic")
    return rgb

def frequency_domain_demosaic(raw, config):
    """频域demosaic - 减少高频伪影"""
    # 先用EA算法
    rgb = opencv_demosaic_ea(raw, config)
    
    # 频域处理每个通道
    for c in range(3):
        channel = rgb[:,:,c]
        
        # FFT
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        
        # 创建低通滤波器（抑制高频摩尔纹）
        h, w = channel.shape
        crow, ccol = h//2, w//2
        
        # 高斯低通滤波器
        y, x = np.ogrid[:h, :w]
        mask = np.exp(-((x - ccol)**2 + (y - crow)**2) / (2 * (min(h,w)/8)**2))
        
        # 保留低频，轻微抑制高频
        mask = 0.3 + 0.7 * mask  # 不完全滤除高频
        
        # 应用滤波器
        f_shift_filtered = f_shift * mask
        
        # 逆FFT
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        channel_filtered = np.fft.ifft2(f_ishift)
        rgb[:,:,c] = np.real(channel_filtered)
    
    print("DEBUG: 应用频域抗摩尔纹处理")
    return rgb

def interpolate_missing_pixels(rgb, raw, r_mask, g_mask, b_mask):
    """智能插值缺失像素"""
    h, w = raw.shape
    
    # 处理R通道缺失像素
    missing_r = ~r_mask
    if np.any(missing_r):
        # 使用周围G像素估计
        g_avg = cv2.GaussianBlur(rgb[:,:,1], (3,3), 0.8)
        rgb[:,:,0][missing_r] = g_avg[missing_r]
    
    # 处理B通道缺失像素  
    missing_b = ~b_mask
    if np.any(missing_b):
        g_avg = cv2.GaussianBlur(rgb[:,:,1], (3,3), 0.8)
        rgb[:,:,2][missing_b] = g_avg[missing_b]
    
    # 处理G通道缺失像素
    missing_g = ~g_mask
    if np.any(missing_g):
        rb_avg = (rgb[:,:,0] + rgb[:,:,2]) / 2
        rgb[:,:,1][missing_g] = rb_avg[missing_g]
    
    return rgb

def create_bayer_masks(h, w, pattern):
    """创建Bayer模式mask"""
    r_mask = np.zeros((h, w), dtype=bool)
    g_mask = np.zeros((h, w), dtype=bool) 
    b_mask = np.zeros((h, w), dtype=bool)
    
    if pattern == "rggb":
        r_mask[0::2, 0::2] = True
        g_mask[0::2, 1::2] = True
        g_mask[1::2, 0::2] = True
        b_mask[1::2, 1::2] = True
    elif pattern == "bggr":
        b_mask[0::2, 0::2] = True
        g_mask[0::2, 1::2] = True
        g_mask[1::2, 0::2] = True
        r_mask[1::2, 1::2] = True
    
    return r_mask, g_mask, b_mask

def fill_borders(rgb, raw, r_mask, g_mask, b_mask):
    """填充边界像素"""
    h, w = raw.shape
    
    for y in [0, 1, h-2, h-1]:
        for x in range(w):
            for c in range(3):
                if rgb[y, x, c] == 0:
                    for dy in [-2, -1, 0, 1, 2]:
                        for dx in [-2, -1, 0, 1, 2]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and rgb[ny, nx, c] > 0:
                                rgb[y, x, c] = rgb[ny, nx, c]
                                break
                        if rgb[y, x, c] > 0:
                            break
    
    return rgb

def bilinear_demosaic(raw, pattern):
    """简单双线性插值demosaic"""
    h, w = raw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    if pattern == "bggr":
        rgb[0::2, 0::2, 2] = raw[0::2, 0::2]  # B
        rgb[0::2, 1::2, 1] = raw[0::2, 1::2]  # G
        rgb[1::2, 0::2, 1] = raw[1::2, 0::2]  # G  
        rgb[1::2, 1::2, 0] = raw[1::2, 1::2]  # R
    
    # 双线性插值填充缺失像素
    for c in range(3):
        mask = rgb[:,:,c] == 0
        if np.any(mask):
            from scipy.interpolate import griddata
            y_known, x_known = np.where(~mask)
            values_known = rgb[y_known, x_known, c]
            y_missing, x_missing = np.where(mask)
            
            if len(y_known) > 0:
                interpolated = griddata(
                    (y_known, x_known), values_known,
                    (y_missing, x_missing), method='linear',
                    fill_value=0
                )
                rgb[y_missing, x_missing, c] = interpolated
    
    return rgb

def edge_aware_demosaic(raw, pattern):
    """边缘感知demosaic"""
    rgb = bilinear_demosaic(raw, pattern)
    
    gray = np.mean(rgb, axis=2)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    edge_mask = edges > 0
    
    return rgb

def opencv_demosaic(raw, config):
    """OpenCV demosaic方法 - 支持所有Bayer模式"""
    pattern = config.get("bayer_pattern", "bggr").lower()
    
    print(f"DEBUG: opencv_demosaic接收到的pattern: '{pattern}'")
    print(f"DEBUG: config内容: {config}")
    
    # 归一化到8bit进行demosaic
    raw_8bit = (raw / raw.max() * 255).astype(np.uint8)
    
    # 根据Bayer模式选择对应的OpenCV转换
    if pattern == "bggr":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerBG2RGB_VNG)
        print("DEBUG: 使用 COLOR_BayerBG2RGB_VNG")
    elif pattern == "grbg":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerGR2RGB_VNG)
        print("DEBUG: 使用 COLOR_BayerGR2RGB_VNG")
    elif pattern == "rggb":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerRG2RGB_VNG)
        print("DEBUG: 使用 COLOR_BayerRG2RGB_VNG")
    elif pattern == "gbrg":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerGB2RGB_VNG)
        print("DEBUG: 使用 COLOR_BayerGB2RGB_VNG")
    else:
        print(f"WARNING: 未知Bayer模式 '{pattern}', 默认使用BGGR")
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerBG2RGB_VNG)
    
    # 转回float32，保持在合理范围
    rgb_float = rgb_8bit.astype(np.float32) * (raw.max() / 255.0)
    
    return rgb_float

def opencv_demosaic_ea(raw, config):
    """OpenCV边缘感知demosaic - 减少摩尔纹"""
    pattern = config.get("bayer_pattern", "bggr").lower()
    
    # 归一化到8bit
    raw_8bit = (raw / raw.max() * 255).astype(np.uint8)
    
    # 使用边缘感知算法
    if pattern == "bggr":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerBG2RGB_EA)
    elif pattern == "grbg":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerGR2RGB_EA)
    elif pattern == "rggb":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerRG2RGB_EA)
    elif pattern == "gbrg":
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerGB2RGB_EA)
    else:
        rgb_8bit = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerBG2RGB_EA)
    
    print("DEBUG: 使用边缘感知demosaic (EA)")
    
    # 转回float32
    rgb_float = rgb_8bit.astype(np.float32) * (raw.max() / 255.0)
    
    return rgb_float

def anti_moire_demosaic(raw, config):
    """抗摩尔纹demosaic算法"""
    pattern = config.get("bayer_pattern", "bggr").lower()
    
    # 先用边缘感知demosaic
    rgb = opencv_demosaic_ea(raw, config)
    
    # 检测高频区域（容易产生摩尔纹的区域）
    gray = cv2.cvtColor((rgb * 255 / rgb.max()).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 高频检测
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    high_freq_mask = np.abs(laplacian) > np.percentile(np.abs(laplacian), 85)
    
    # 在高频区域应用轻微的低通滤波
    if np.any(high_freq_mask):
        # 创建3通道mask
        mask_3d = np.stack([high_freq_mask] * 3, axis=2)
        
        # 轻微高斯模糊
        rgb_blurred = cv2.GaussianBlur(rgb, (3, 3), 0.8)
        
        # 只在高频区域混合
        alpha = 0.3  # 混合系数
        rgb = np.where(mask_3d, 
                      rgb * (1 - alpha) + rgb_blurred * alpha, 
                      rgb)
    
    print("DEBUG: 应用抗摩尔纹处理")
    
    return rgb

def selective_anti_moire_demosaic(raw, config):
    """选择性抗摩尔纹 - 保持细节"""
    # 使用最佳的OpenCV算法
    rgb = opencv_demosaic_ea(raw, config)
    
    # 检测摩尔纹区域（周期性高频模式）
    gray = cv2.cvtColor((rgb * 255 / rgb.max()).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 使用方向性滤波器检测摩尔纹
    kernel_45 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3
    kernel_135 = np.array([[1, 1, -1], [1, 0, -1], [-1, -1, 1]]) / 3
    
    response_45 = cv2.filter2D(gray, cv2.CV_32F, kernel_45)
    response_135 = cv2.filter2D(gray, cv2.CV_32F, kernel_135)
    
    # 摩尔纹通常表现为强烈的方向性响应
    moire_strength = np.abs(response_45) + np.abs(response_135)
    moire_mask = moire_strength > np.percentile(moire_strength, 95)
    
    # 只在摩尔纹区域应用极轻微的处理
    if np.any(moire_mask):
        # 非常轻微的各向异性扩散
        for c in range(3):
            channel = rgb[:,:,c]
            # 只在摩尔纹像素上应用1次轻微平滑
            smoothed = cv2.bilateralFilter(
                (channel * 255).astype(np.uint8), 
                d=3, sigmaColor=10, sigmaSpace=10
            ).astype(np.float32) / 255 * rgb.max()
            
            # 极小的混合比例
            alpha = 0.15
            rgb[:,:,c] = np.where(moire_mask, 
                                 channel * (1-alpha) + smoothed * alpha,
                                 channel)
    
    print("DEBUG: 选择性抗摩尔纹 - 保持细节")
    return rgb
