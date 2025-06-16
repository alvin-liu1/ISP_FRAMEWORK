
# ---------------------
# 锐化模块
# ✅ 通常放在 ISP 流程末尾，如 Gamma 之后或之前，增强图像细节。

import cv2
import numpy as np

def apply(rgb, config):
    # 确保输入 rgb 是 0-1 范围的 float32
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    sharpen_method = config.get("method", "unsharp_masking") # 默认非锐化掩蔽

    # 将 0-1 浮点数临时缩放到 0-255 uint8，以便 OpenCV 函数处理
    rgb_8bit = (rgb * 255.0).astype(np.uint8)

    sharpened_rgb_8bit = np.copy(rgb_8bit) # 初始化锐化后的图像

    if sharpen_method == "unsharp_masking":
        # 非锐化掩蔽
        blur_kernel_size = config.get("blur_kernel_size", 5) # 用于生成模糊图像的核大小，必须是奇数
        sharpen_strength = config.get("strength", 1.0)        # 锐化强度，越大越锐利
        
        # 确保核大小是奇数
        if blur_kernel_size % 2 == 0:
            print(f"警告: 非锐化掩蔽模糊核大小 {blur_kernel_size} 必须是奇数，已调整为 {blur_kernel_size + 1}。")
            blur_kernel_size += 1

        # 1. 生成模糊图像
        blurred_rgb_8bit = cv2.GaussianBlur(rgb_8bit, (blur_kernel_size, blur_kernel_size), 0)

        # 2. 计算细节（原始图像 - 模糊图像）
        # 这里需要将 uint8 转换为 float，否则减法可能产生负值，且细节信息被截断
        details = rgb_8bit.astype(np.float32) - blurred_rgb_8bit.astype(np.float32)

        # 3. 将细节按强度加回原始图像
        # 注意：这里我们加到原始的 0-255 8bit 图像的浮点版本上
        sharpened_rgb_float = rgb_8bit.astype(np.float32) + details * sharpen_strength
        
        # 4. 裁剪并转换回 8bit
        sharpened_rgb_8bit = np.clip(sharpened_rgb_float, 0, 255).astype(np.uint8)
        print(f"应用锐化 (非锐化掩蔽), 模糊核大小: {blur_kernel_size}, 强度: {sharpen_strength}")

    # 你可以根据需要添加其他锐化方法，例如拉普拉斯锐化，但通常不推荐单独使用。
    # elif sharpen_method == "laplacian":
    #     kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    #     sharpened_rgb_8bit = cv2.filter2D(rgb_8bit, -1, kernel)
    #     print(f"应用锐化 (拉普拉斯)")

    else:
        print(f"警告: 未知的锐化方法 '{sharpen_method}'。未应用锐化。")
        sharpened_rgb_8bit = rgb_8bit # 如果方法未知，返回原始图像

    # 将处理后的 8 位图像重新转换回 0-1 范围的 float32
    sharpened_rgb = sharpened_rgb_8bit.astype(np.float32) / 255.0
    
    # 裁剪到 0-1 范围，防止转换误差
    return np.clip(sharpened_rgb, 0, 1)