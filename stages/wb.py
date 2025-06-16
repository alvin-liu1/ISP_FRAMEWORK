# stages/wb.py
import numpy as np

def apply(rgb, config):
    # 确保输入 rgb 是 float32 类型，并裁剪到 0-1 范围
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    
    wb_method = config.get("method", "manual")
    
    if wb_method == "manual":
        # 手动设置增益
        gains = config.get("gains", [1.0, 1.0, 1.0])
        r_gain, g_gain, b_gain = gains
    elif wb_method == "gray_world":
        # 灰度世界算法
        # 计算每个通道的平均值
        avg_r, avg_g, avg_b = np.mean(rgb, axis=(0, 1))
        
        # 避免除以零或非常小的值
        avg_r = max(avg_r, 1e-6)
        avg_g = max(avg_g, 1e-6)
        avg_b = max(avg_b, 1e-6)

        # 以绿色通道为基准进行白平衡（通常效果较好）
        r_gain = avg_g / avg_r
        b_gain = avg_g / avg_b
        g_gain = 1.0 # 绿色通道增益通常设为 1.0
        gains = [r_gain, g_gain, b_gain]
        print(f"AWB (Gray World) gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")

    elif wb_method == "max_rgb":
        # 完美反射 / 最亮像素算法
        # 找到所有通道中像素值总和最大的像素
        # sum_channels = np.sum(rgb, axis=-1)
        # max_idx = np.unravel_index(np.argmax(sum_channels), sum_channels.shape)
        # bright_pixel_r, bright_pixel_g, bright_pixel_b = rgb[max_idx]

        # 更健壮的完美反射：找到亮度最高的像素，并以其最大分量为基准。
        # 避免某个通道饱和，但其他通道很低的情况。
        max_channel_val = np.max(rgb) # 找到图像所有像素中，所有通道的最高值
        
        # 找到所有通道值之和最大的像素，并取其各个通道值
        # 如果图像全黑，max_channel_val 可能为 0，需处理
        if max_channel_val == 0:
            r_gain, g_gain, b_gain = 1.0, 1.0, 1.0 # 全黑图像不进行白平衡
        else:
            # 找到图像中最亮像素（即某个分量达到 max_channel_val 的像素）
            # 找到像素值之和最高的像素（避免某个通道饱和但整体不亮）
            # 假设有一个像素接近白色（R, G, B值都高）
            # 寻找“最白”的像素，通常是 R+G+B 最大的像素
            sum_of_channels = np.sum(rgb, axis=-1)
            # 获取最大值所在的索引，返回 (y, x) 坐标
            max_sum_idx = np.unravel_index(np.argmax(sum_of_channels), sum_of_channels.shape)
            
            # 获取该像素的 R, G, B 值
            bright_pixel_r, bright_pixel_g, bright_pixel_b = rgb[max_sum_idx]
            
            # 避免除以零
            bright_pixel_r = max(bright_pixel_r, 1e-6)
            bright_pixel_g = max(bright_pixel_g, 1e-6)
            bright_pixel_b = max(bright_pixel_b, 1e-6)
            
            # 以最亮的通道为基准 (通常是 max_channel_val)
            # 或以最高平均亮度通道为基准 (比如 G 通道)，这里我们用所有通道中的最大值作为基准
            reference_val = max_channel_val # np.max(rgb)
            
            r_gain = reference_val / bright_pixel_r
            g_gain = reference_val / bright_pixel_g
            b_gain = reference_val / bright_pixel_b
            gains = [r_gain, g_gain, b_gain]
            print(f"AWB (Max RGB) gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")

    else:
        print(f"Warning: Unknown WB method '{wb_method}'. Using default manual gains [1.0, 1.0, 1.0].")
        gains = [1.0, 1.0, 1.0]

    r_gain, g_gain, b_gain = gains
    
    # 应用增益
    rgb[..., 0] *= r_gain
    rgb[..., 1] *= g_gain
    rgb[..., 2] *= b_gain
    
    # 裁剪回 0-1 范围，保持 float32 类型
    return np.clip(rgb, 0, 1).astype(np.float32)