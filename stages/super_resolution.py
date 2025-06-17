import numpy as np
from scipy.ndimage import zoom, gaussian_filter

def apply(rgb, config):
    """
    对 RGB 图像应用超分辨率（放大），使用插值算法，并可选地应用非锐化掩膜以增强感知清晰度。

    参数:
        rgb (np.ndarray): 输入的 RGB 图像数据 (float32 类型，值域通常在 0-1 之间)。
                          假定格式为 [高, 宽, 通道]。
        config (dict): 超分辨率的配置字典，包含以下键：
                       - 'enable' (bool): 是否启用此阶段。
                       - 'scale_factor' (float): 图像的放大因子（例如，2.0 表示 2 倍放大）。
                       - 'upscale_method' (str): 插值方法 ('bicubic' 双三次, 'nearest' 最近邻, 'bilinear' 双线性)。
                                                 'bicubic' 通常推荐用于较好的图像质量。
                       - 'sharpen_enable' (bool): 是否应用非锐化掩膜进行锐化。
                       - 'sharpen_amount' (float): 非锐化掩膜的强度（例如，1.0）。
                       - 'sharpen_radius' (float): 用于非锐化掩膜的模糊半径（例如，1.0）。

    返回:
        np.ndarray: 放大后并可选地经过锐化的 RGB 图像数据 (float32 类型，值域裁剪到 0-1 之间)。
    """
    if not config.get('enable', False):
        print("超分辨率模块已禁用。")
        return rgb

    print("正在应用超分辨率...")
    
    scale_factor = config.get('scale_factor', 2.0)
    upscale_method = config.get('upscale_method', 'bicubic').lower()
    sharpen_enable = config.get('sharpen_enable', False)
    sharpen_amount = config.get('sharpen_amount', 1.0)
    sharpen_radius = config.get('sharpen_radius', 1.0)

    # 验证放大因子
    if scale_factor <= 1.0:
        print(f"警告：超分辨率放大因子 {scale_factor} <= 1.0。未执行放大操作。")
        return rgb
    
    # 为 scipy.ndimage.zoom 选择插值顺序
    # order=0: 最近邻插值
    # order=1: 双线性插值
    # order=3: 双三次（立方）插值
    interpolation_order = 3 # 默认为双三次插值
    if upscale_method == 'nearest':
        interpolation_order = 0
    elif upscale_method == 'bilinear':
        interpolation_order = 1
    elif upscale_method == 'bicubic':
        interpolation_order = 3
    else:
        print(f"警告：未知的放大方法 '{upscale_method}'。回退到双三次插值。")

    # 执行图像放大
    # zoom 函数需要为每个维度（高、宽、通道）提供一个缩放因子
    # 图像通道维度（最后一个维度）的缩放因子为 1，表示不跨通道插值
    upscaled_rgb = zoom(rgb, (scale_factor, scale_factor, 1), order=interpolation_order)

    # 应用非锐化掩膜（用于增强感知清晰度）
    if sharpen_enable:
        # 创建一个模糊版本的图像
        blurred_rgb = np.zeros_like(upscaled_rgb)
        # 逐通道处理，以避免跨通道模糊（例如，不希望红色模糊到绿色通道）
        for i in range(upscaled_rgb.shape[-1]): 
            # gaussian_filter 期望为每个维度提供 sigma 值
            # 我们只在空间上模糊，不在通道上模糊，所以通道的 sigma 为 0
            blurred_rgb[..., i] = gaussian_filter(upscaled_rgb[..., i], sigma=sharpen_radius)
        
        # 计算细节层（原图减去模糊图），细节层包含了图像的边缘信息
        detail_layer = upscaled_rgb - blurred_rgb
        # 将细节层按强度加回到原图，从而增强边缘
        sharpened_rgb = upscaled_rgb + detail_layer * sharpen_amount
        
        # 裁剪值域以确保像素值保持在有效范围 [0, 1]
        final_rgb = np.clip(sharpened_rgb, 0, 1)
    else:
        final_rgb = upscaled_rgb

    # 确保最终输出的像素值被裁剪到 0-1 范围，因为这通常是显示/保存前的最后一步
    return np.clip(final_rgb, 0, 1)