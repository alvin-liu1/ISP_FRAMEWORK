# stages/denoise.py
# ---------------------
# 去噪模块
# ✅ 放在 ISP 流程中较早的位置（例如 WB 之后，CCM 之前），去除随机噪声。

import cv2
import numpy as np

def apply(rgb, config):
    # 确保输入 rgb 是 0-1 范围的 float32
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    denoise_method = config.get("method", "gaussian") # 默认使用高斯模糊
    
    # 将 0-1 浮点数临时缩放到 0-255 uint8，以便 OpenCV 函数处理
    # 大多数 OpenCV 滤波函数针对 uint8/uint16 优化
    rgb_8bit = (rgb * 255.0).astype(np.uint8)

    denoised_rgb_8bit = np.copy(rgb_8bit) # 初始化去噪后的图像

    if denoise_method == "gaussian":
        # 高斯模糊
        kernel_size = config.get("kernel_size", 5) # 模糊核大小，必须是奇数
        sigma_x = config.get("sigma_x", 0)         # X 方向标准差，0 表示根据核大小计算
        
        # 确保核大小是奇数
        if kernel_size % 2 == 0:
            print(f"警告: 高斯模糊核大小 {kernel_size} 必须是奇数，已调整为 {kernel_size + 1}。")
            kernel_size += 1

        # cv2.GaussianBlur 期望 BGR 顺序
        denoised_rgb_8bit = cv2.GaussianBlur(denoised_rgb_8bit, (kernel_size, kernel_size), sigma_x)
        print(f"应用去噪 (高斯模糊), 核大小: {kernel_size}, SigmaX: {sigma_x}")

    elif denoise_method == "bilateral":
        # 双边滤波
        # d: 像素邻域直径，越大处理速度越慢但考虑范围越大
        d = config.get("diameter", 9) 
        # sigma_color: 颜色空间标准差。越大，颜色相差大的像素也会被平均
        sigma_color = config.get("sigma_color", 50) 
        # sigma_space: 坐标空间标准差。越大，越远的像素也会被平均
        sigma_space = config.get("sigma_space", 50) 

        # cv2.bilateralFilter 期望 BGR 顺序
        denoised_rgb_8bit = cv2.bilateralFilter(denoised_rgb_8bit, d, sigma_color, sigma_space)
        print(f"应用去噪 (双边滤波), Diameter: {d}, SigmaColor: {sigma_color}, SigmaSpace: {sigma_space}")

    elif denoise_method == "median":
        # 中值滤波
        kernel_size = config.get("kernel_size", 5) # 核大小，必须是奇数
        if kernel_size % 2 == 0:
            print(f"警告: 中值滤波核大小 {kernel_size} 必须是奇数，已调整为 {kernel_size + 1}。")
            kernel_size += 1
        
        # cv2.medianBlur 对多通道图像按通道独立处理
        denoised_rgb_8bit = cv2.medianBlur(denoised_rgb_8bit, kernel_size)
        print(f"应用去噪 (中值滤波), 核大小: {kernel_size}")
    
    elif denoise_method == "nl_means": # 添加非局部均值去噪
        # h: 决定滤波器强度的参数。较大的 h 值可以更好地去除噪声，但也会导致更多的细节丢失。
        # h_color: 与 h 相似，但用于颜色分量。
        # templateWindowSize: 用于计算权重的方形模板窗口大小。推荐值 7。
        # searchWindowSize: 用于搜索相似补丁的方形窗口大小。推荐值 21。
        h_param = config.get("h_param", 10) # 建议值通常在 5-20 之间，具体取决于噪声水平
        h_color_param = config.get("h_color_param", h_param) # 颜色参数，通常与 h_param 相同
        template_window_size = config.get("template_window_size", 7)
        search_window_size = config.get("search_window_size", 21)

        # cv2.fastNlMeansDenoisingColored 专门用于彩色图像
        denoised_rgb_8bit = cv2.fastNlMeansDenoisingColored(
            rgb_8bit, 
            None, # dst 参数，通常设为 None
            h_param, 
            h_color_param, 
            template_window_size, 
            search_window_size
        )
        print(f"应用去噪 (非局部均值), h: {h_param}, h_color: {h_color_param}, template_ws: {template_window_size}, search_ws: {search_window_size}")


    else:
        print(f"警告: 未知的去噪方法 '{denoise_method}'。未应用去噪。")
        denoised_rgb_8bit = rgb_8bit # 如果方法未知，返回原始图像

    # 将处理后的 8 位图像重新转换回 0-1 范围的 float32
    denoised_rgb = denoised_rgb_8bit.astype(np.float32) / 255.0
    
    # 裁剪到 0-1 范围，防止转换误差
    return np.clip(denoised_rgb, 0, 1)