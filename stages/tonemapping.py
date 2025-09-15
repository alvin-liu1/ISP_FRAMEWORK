import numpy as np

def apply(rgb, config):
    """
    Apply tone mapping to the RGB image.
    This version is further optimized to prevent excessive darkening of normal brightness inputs,
    while still allowing for shadow lift, highlight compression, and contrast control.

    Args:
        rgb (np.ndarray): Input RGB image data (float32, values potentially > 1.0 for HDR).
        config (dict): Configuration dictionary for tonemapping, containing:
                       - 'lift' (float): Controls shadow/dark area lifting (overall exposure).
                       - 'roll' (float): Controls the point where highlight compression begins/intensifies.
                                         Higher values reduce compression for normal inputs.
                       - 'compress' (float): Controls the strength of highlight compression.
                                             Lower values reduce compression strength.
                       - 'contrast' (float): Controls the mid-tone contrast.
                       - 'brightness' (float): Overall brightness multiplier after tone mapping.

    Returns:
        np.ndarray: Tone-mapped RGB image data (float32, values clipped to 0-1 range).
    """
    # Ensure input is float32
    rgb = rgb.astype(np.float32)

    # --- 从 config 字典中获取参数 ---
    lift = config.get("lift", 0.2)     # 默认值，用于整体曝光提升
    roll = config.get("roll", 0.8)     # 默认值，调整高光压缩的起始点 (关键参数)
    compress = config.get("compress", 0.3) # 默认值，调整高光压缩的强度 (关键参数)
    contrast = config.get("contrast", 1.0) # 默认值
    brightness = config.get("brightness", 1.0) # 默认值
    # --- 参数获取结束 ---

    # 确保 RGB 值大于 0，避免数学问题（如 log(0) 或除以零）
    rgb_safe = np.maximum(rgb, 1e-6)

    # 将 RGB 转换为亮度 Y (ITU-R BT.709 标准亮度系数)
    Y = 0.2126 * rgb_safe[:,:,0] + 0.7152 * rgb_safe[:,:,1] + 0.0722 * rgb_safe[:,:,2]

    # --- 核心优化：多阶段亮度调整 ---

    # 1. 曝光增益 (Exposure Gain): 主要受 lift 参数控制，用于整体提亮，特别是暗部。
    # 保持 lift 的显著效果，确保即使 Y 较低，也能得到足够的 Y_exposed。
    exposure_gain = 1.0 + lift * 10.0 
    Y_exposed = Y * exposure_gain

    # 2. 对比度调整 (Contrast Adjustment): 通过伽马曲线来控制中间调的对比度。
    contrast_gamma = 1.0 / (contrast + 1e-6) 
    Y_contrasted = np.power(Y_exposed, contrast_gamma)

    # 3. Filmic S-curve 核心映射：处理大动态范围，特别是高光压缩。
    # *** 关键修改：调整 high_compression_point 和 compression_strength 的计算 ***

    # high_compression_point (高光压缩的枢轴点):
    # 确保此点默认情况下远高于 1.0。这意味着对于大部分“正常”亮度值 (0-1 或 0-2)，
    # 曲线不会过早地进入强压缩区域，从而避免整体变暗。
    # roll 的值越大，这个点越高，高光压缩开始得越晚，图像整体越亮。
    high_compression_point = 1.0 + roll * 5.0 # `roll` 乘以 5.0，确保 `high_compression_point` 默认足够高 (例如 1 + 0.8*5 = 5.0)

    # compression_strength (高光压缩的强度):
    # 降低默认强度，避免过度压缩正常亮度。
    # compress 的值越小，强度越低，高光越亮。
    compression_strength = 0.2 + compress * 2.0 # 基础值设为 0.2，并降低 `compress` 的乘数

    # 应用 S-curve 公式
    denominator = 1.0 + Y_contrasted / (high_compression_point + 1e-6) * (compression_strength + 1e-6)
    mapped_Y = Y_contrasted / denominator
    
    # 4. 整体亮度乘数：最终调整整体图像亮度。
    mapped_Y = mapped_Y * brightness

    # --- 多阶段调整结束 ---

    # 保持色度，只调整亮度 (将亮度的变化比例重新应用回 RGB 分量)
    scale_factor = np.divide(mapped_Y, Y, out=np.zeros_like(Y), where=Y!=0)
    
    # 将亮度调整因子应用到每个 RGB 通道
    rgb_tonemapped = rgb * np.stack([scale_factor, scale_factor, scale_factor], axis=-1)

    # 修改：不要过早裁剪，保留一定的超出范围
    preserve_headroom = config.get("preserve_headroom", False)
    if preserve_headroom:
        # 软裁剪：保留一些超出1.0的值，让后续Gamma处理
        max_value = config.get("max_output_value", 1.2)
        rgb_tonemapped = np.clip(rgb_tonemapped, 0, max_value)
        print(f"Tonemapping: 保留headroom，最大值: {max_value}")
    else:
        # 原始硬裁剪
        rgb_tonemapped = np.clip(rgb_tonemapped, 0, 1)
    
    return rgb_tonemapped
