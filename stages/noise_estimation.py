import numpy as np
from scipy import ndimage

def estimate_noise_level(raw, config):
    """估计图像噪声水平，用于后续自适应处理"""
    # 使用Laplacian算子估计噪声
    laplacian = ndimage.laplace(raw.astype(np.float32))
    noise_variance = np.var(laplacian) / 6.0  # 理论系数
    
    # 基于暗区域的噪声估计
    dark_threshold = config.get("dark_threshold", 0.1)
    dark_mask = raw < dark_threshold
    if np.sum(dark_mask) > 100:  # 确保有足够的暗像素
        dark_noise = np.std(raw[dark_mask])
        noise_variance = max(noise_variance, dark_noise**2)
    
    # 新增：基于梯度的噪声估计（更准确）
    sobel_x = ndimage.sobel(raw, axis=1)
    sobel_y = ndimage.sobel(raw, axis=0)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 选择低梯度区域进行噪声估计
    low_gradient_threshold = np.percentile(gradient_magnitude, 20)
    low_gradient_mask = gradient_magnitude < low_gradient_threshold
    
    if np.sum(low_gradient_mask) > 1000:
        gradient_noise = np.std(raw[low_gradient_mask])
        noise_variance = max(noise_variance, gradient_noise**2)
    
    return np.sqrt(noise_variance)

def apply(raw, config):
    """自适应噪声估计"""
    noise_level = estimate_noise_level(raw, config)
    
    # 将噪声水平存储到config中供后续模块使用
    config['estimated_noise_level'] = noise_level
    
    # 根据噪声水平调整后续模块参数
    if noise_level > 0.05:  # 高噪声
        config.setdefault('denoise', {})['enable'] = True
        config['denoise']['h_param'] = min(15, max(8, int(noise_level * 200)))
    elif noise_level > 0.02:  # 中等噪声
        config.setdefault('denoise', {})['h_param'] = min(10, max(5, int(noise_level * 150)))
    
    print(f"估计噪声水平: {noise_level:.4f}")
    return raw
