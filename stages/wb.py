import numpy as np

def apply(rgb, config):
    # 确保输入 rgb 是 float32 类型，并裁剪到 0-1 范围
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    wb_method = config.get("method", "manual")

    # --- 新增：定义亮度阈值 ---
    # 这些值需要根据你的图像内容和经验进行调整
    # 例如，排除最暗的 5% 和最亮的 1% 像素
    min_luminance_threshold = config.get("wb_min_luminance_threshold", 0.05) # 比如 5%
    max_luminance_threshold = config.get("wb_max_luminance_threshold", 0.99) # 比如 99%

    # 计算亮度（例如使用Y通道或平均RGB）
    # 这里使用简单的平均RGB作为亮度估计，更精确的可以用Y通道（Luma = 0.2126*R + 0.7152*G + 0.0722*B）
    luminance = np.mean(rgb, axis=-1) # 形状为 (H, W)

    # 创建有效像素掩码：亮度在阈值范围内的像素
    valid_pixels_mask = (luminance > min_luminance_threshold) & (luminance < max_luminance_threshold)

    # 检查是否有足够的有效像素
    if not np.any(valid_pixels_mask):
        print("Warning: No valid pixels found for white balance calculation. Using default gains.")
        r_gain, g_gain, b_gain = 1.0, 1.0, 1.0
        gains = [r_gain, g_gain, b_gain]
    else:
        # 提取有效像素的RGB值
        valid_rgb_pixels = rgb[valid_pixels_mask] # 形状为 (N, 3)，N是有效像素的数量

        if wb_method == "manual":
            gains = config.get("gains", [1.0, 1.0, 1.0])
            r_gain, g_gain, b_gain = gains
        elif wb_method == "gray_world":
            # 灰度世界算法
            # 现在只在有效像素上计算平均值
            avg_r, avg_g, avg_b = np.mean(valid_rgb_pixels, axis=0) # 对N个像素求平均

            # 避免除以零或非常小的值
            avg_r = max(avg_r, 1e-6)
            avg_g = max(avg_g, 1e-6)
            avg_b = max(avg_b, 1e-6)

            r_gain = avg_g / avg_r
            b_gain = avg_g / avg_b
            g_gain = 1.0
            gains = [r_gain, g_gain, b_gain]
            print(f"AWB (Gray World - filtered) gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")

        elif wb_method == "max_rgb":
            # 完美反射 / 最亮像素算法
            # 在有效像素中寻找最亮的
            # 寻找“最白”的像素，通常是 R+G+B 最大的像素
            sum_of_channels_valid = np.sum(valid_rgb_pixels, axis=-1) # (N,)

            # 如果有效像素都是黑的
            if np.max(sum_of_channels_valid) == 0:
                r_gain, g_gain, b_gain = 1.0, 1.0, 1.0 # 全黑图像不进行白平衡
            else:
                max_sum_idx_in_valid = np.argmax(sum_of_channels_valid)
                bright_pixel_r, bright_pixel_g, bright_pixel_b = valid_rgb_pixels[max_sum_idx_in_valid]

                bright_pixel_r = max(bright_pixel_r, 1e-6)
                bright_pixel_g = max(bright_pixel_g, 1e-6)
                bright_pixel_b = max(bright_pixel_b, 1e-6)

                # 以所有有效像素中，所有通道的最高值作为基准
                reference_val = np.max(valid_rgb_pixels) # 或 np.max(bright_pixel_r, bright_pixel_g, bright_pixel_b)
                if reference_val == 0: # 再次检查避免除零
                    r_gain, g_gain, b_gain = 1.0, 1.0, 1.0
                else:
                    r_gain = reference_val / bright_pixel_r
                    g_gain = reference_val / bright_pixel_g
                    b_gain = reference_val / bright_pixel_b
                gains = [r_gain, g_gain, b_gain]
                print(f"AWB (Max RGB - filtered) gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")

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