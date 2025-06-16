# --------------------- 
# # 黑电平校正模块（Black Level Correction） 
# # ✅ 通常应在 demosaic 之前做，用于去除传感器底噪。
"""import numpy as np
from utils.image_io import save_image_debug

def apply(raw, config, debug_path=None):
    black_level = config.get("black_level",64)
    
    if debug_path:         
         save_image_debug(raw, debug_path.replace(".png", "_before.png"), scale=True)
    corrected = raw - black_level
    corrected = np.clip(corrected, 0, None)
    if debug_path:         
        save_image_debug(corrected, debug_path.replace(".png", "_after.png"), scale=True)
    
    return corrected
    """

import numpy as np
# 假设 utils.image_io.save_image_debug 模块已存在
# from utils.image_io import save_image_debug 

def apply(raw, config, debug_path=None):
    # --- 优化点 1: 获取传感器位深，以便更准确地处理数据范围 ---
    # 允许在 config 中指定传感器的位深，如果没有则默认为 10 位。
    # 这对于后续的数值解释和归一化至关重要。
    sensor_bit_depth = config.get("sensor_bit_depth", 10) 
    max_sensor_value = (2**sensor_bit_depth) - 1

    # --- 优化点 2: 处理原始数据类型，确保在 uint16 上操作 ---
    # 假设 raw 数据在进入 BLC 模块时，已经是最接近传感器输出的原始整数值。
    # 如果 raw 是 float 类型（例如 0-1 范围的），这里需要将其缩放回原始位深范围的整数。
    # 为了保持函数名和变量名不变，我们仍然使用 raw_for_blc 来表示这个准备好的数据。
    # 这里假设 raw 已经是 uint16 或可以安全地转换为 uint16。
    # 如果 raw 是 float，通常需要根据 max_sensor_value 进行缩放后再转 uint16。
    # 例如：raw_for_blc = (raw * max_sensor_value).astype(np.uint16)
    raw_for_blc = raw.astype(np.uint16) # 确保数据为 uint16 类型

    # --- 调试可视化（如果需要）---
    if debug_path:
        # save_image_debug 函数可能需要调整，以使用 max_sensor_value 进行适当的缩放，
        # 从而使调试图像在不同位深下都能正确显示亮度。
        # save_image_debug(raw_for_blc, debug_path.replace(".png", "_before.png"), scale=True, max_val_for_scale=max_sensor_value)
        pass # 注释掉实际调用，因为它依赖外部模块

    # --- 优化点 3: 支持通道独立的黑电平校正（更精确） ---
    # 检查 config 中是否有 'black_level_channels'。
    # 这是一个字典，例如：{"R": 60, "Gr": 62, "Gb": 61, "B": 65}。
    # 如果有，则按通道减去黑电平；否则，使用通用的 'black_level'。
    black_levels_per_channel = config.get("black_level_channels", None)
    pattern = config.get("bayer_pattern", "rggb").lower() # 获取拜耳模式

    # corrected 变量将存储校正后的结果
    corrected = np.copy(raw_for_blc) # 先复制一份，避免直接修改原始 raw_for_blc

    if black_levels_per_channel and pattern:
        # 根据拜耳模式和通道黑电平进行校正
        # 这里以 RGGB 为例，你需要根据实际支持的模式扩展
        if pattern == "rggb":
            # R 通道 (0,0), (0,2) ...
            corrected[0::2, 0::2] = np.clip(raw_for_blc[0::2, 0::2] - black_levels_per_channel.get("R", 0), 0, None)
            # Gr 通道 (0,1), (0,3) ...
            corrected[0::2, 1::2] = np.clip(raw_for_blc[0::2, 1::2] - black_levels_per_channel.get("Gr", 0), 0, None)
            # Gb 通道 (1,0), (1,2) ...
            corrected[1::2, 0::2] = np.clip(raw_for_blc[1::2, 0::2] - black_levels_per_channel.get("Gb", 0), 0, None)
            # B 通道 (1,1), (1,3) ...
            corrected[1::2, 1::2] = np.clip(raw_for_blc[1::2, 1::2] - black_levels_per_channel.get("B", 0), 0, None)
        elif pattern == "bggr":
            # 扩展 BGGR, GRBG, GBRG 模式的通道索引和对应的黑电平
            corrected[0::2, 0::2] = np.clip(raw_for_blc[0::2, 0::2] - black_levels_per_channel.get("B", 0), 0, None)
            corrected[0::2, 1::2] = np.clip(raw_for_blc[0::2, 1::2] - black_levels_per_channel.get("Gb", 0), 0, None)
            corrected[1::2, 0::2] = np.clip(raw_for_blc[1::2, 0::2] - black_levels_per_channel.get("Gr", 0), 0, None)
            corrected[1::2, 1::2] = np.clip(raw_for_blc[1::2, 1::2] - black_levels_per_channel.get("R", 0), 0, None)
        # TODO: 根据需要添加其他拜耳模式（grbg, gbrg）
    else:
        # 如果没有提供通道黑电平，则回退到通用的 black_level
        black_level = config.get("black_level", 64) # 默认值 64
        corrected = raw_for_blc - black_level
        corrected = np.clip(corrected, 0, None) # 裁剪到 0，因为物理亮度不能为负

    # --- 调试可视化（如果需要）---
    if debug_path:
        # save_image_debug(corrected, debug_path.replace(".png", "_after.png"), scale=True, max_val_for_scale=max_sensor_value)
        pass # 注释掉实际调用

    # 返回校正后的数组，通常保持为 uint16 格式，因为后续的 ISP 模块（如去马赛克）
    # 更喜欢这种数据类型，以保持精度。
    return corrected
