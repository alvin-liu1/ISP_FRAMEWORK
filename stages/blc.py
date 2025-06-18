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
    # --- 获取传感器位深，以便更准确地处理数据范围 ---
    sensor_bit_depth = config.get("sensor_bit_depth", 10) 
    max_sensor_value = (2**sensor_bit_depth) - 1

    # --- 关键修改点：处理原始数据类型 ---
    # 根据你的DEBUG信息，raw 已经是 float32 且范围在 61.0 到 1023.0。
    # 这意味着它已经代表了原始的 10-bit 整数值，只是以浮点数形式存储。
    # 在这种情况下，我们不应该再乘以 max_sensor_value。
    
    raw_for_blc = raw # 初始赋值

    if raw.dtype == np.float32 or raw.dtype == np.float64:
        # 如果输入是浮点数，且其值范围已经与传感器位深的最大值匹配 (例如，最大值接近 max_sensor_value)
        # 那么我们只需要进行类型转换，而不需要额外缩放。
        # 我们可以添加一个检查，判断是否需要缩放。
        
        # 简单判断是否已经处于 0-1 范围，如果是则需要缩放，否则直接转 uint16
        # 这个判断可能不够严谨，更可靠的方式是明确约定输入raw的范围。
        
        # 假定：如果原始的float32的最大值接近max_sensor_value，那么它就是已经缩放过的，直接转uint16
        # 否则，如果最大值接近1.0，它就是0-1范围的，需要乘以max_sensor_value
        
        # 为了更健壮，最好让调用方明确指定raw的范围，或者在config中添加一个参数
        # 比如 config['input_raw_range'] = '0-1' 或 '0-max_val'
        
        # 针对你提供的DEBUG信息 (最小值: 61.0, 最大值: 1023.0)，
        # 最直接的判断就是它已经处于0-max_sensor_value这个范围了。
        
        if np.max(raw) <= 1.0 + np.finfo(float).eps and np.min(raw) >= 0.0 - np.finfo(float).eps:
            # 如果是0-1范围的浮点数，才进行缩放
            raw_for_blc = (raw * max_sensor_value).astype(np.uint16)
            print(f"DEBUG: BLC: Input raw was float ({raw.dtype}) in 0-1 range, converted to uint16 after scaling by {max_sensor_value}.")
        else:
            # 否则，假设其值已经代表了原始整数值，直接转换为uint16
            raw_for_blc = raw.astype(np.uint16)
            print(f"DEBUG: BLC: Input raw was float ({raw.dtype}) in 0-MAX_VAL range, directly converted to uint16.")
            
    elif raw.dtype == np.uint8:
        # 如果是uint8，通常假设需要从0-255缩放到0-max_sensor_value
        raw_for_blc = (raw.astype(np.float32) * (max_sensor_value / 255.0)).astype(np.uint16)
        print(f"DEBUG: BLC: Input raw was uint8, scaled and converted to uint16.")
    elif raw.dtype == np.uint16:
        # 如果已经是 uint16，直接使用
        raw_for_blc = raw
        print(f"DEBUG: BLC: Input raw was already uint16.")
    else:
        # 对于其他未预期的类型，进行转换并警告
        print(f"WARNING: BLC: Unexpected input raw data type {raw.dtype}. Attempting conversion to uint16.")
        raw_for_blc = raw.astype(np.uint16)

    # ... (后续的 BLC 逻辑保持不变，因为它们是对 uint16 进行操作，且之前已经提升到 int32 避免溢出)

    # --- 调试可视化（如果需要）---
    # if debug_path:
    #     save_image_debug(raw_for_blc, debug_path.replace(".png", "_blc_before.png"), 
    #                      scale=True, max_val_for_scale=max_sensor_value)
    #     print(f"DEBUG: BLC: Saved image before BLC to {debug_path.replace('.png', '_blc_before.png')}")

    # --- 支持通道独立的黑电平校正（更精确） ---
    black_levels_per_channel = config.get("black_level_channels", None)
    pattern = config.get("bayer_pattern", "rggb").lower() 

    corrected = np.copy(raw_for_blc) 

    if black_levels_per_channel and pattern:
        print(f"DEBUG: BLC: Applying channel-wise black level correction for pattern '{pattern}'.")
        if pattern == "rggb":
            r_bl = black_levels_per_channel.get("R", 0)
            corrected[0::2, 0::2] = np.clip(raw_for_blc[0::2, 0::2].astype(np.int32) - r_bl, 0, None).astype(np.uint16)
            gr_bl = black_levels_per_channel.get("Gr", 0)
            corrected[0::2, 1::2] = np.clip(raw_for_blc[0::2, 1::2].astype(np.int32) - gr_bl, 0, None).astype(np.uint16)
            gb_bl = black_levels_per_channel.get("Gb", 0)
            corrected[1::2, 0::2] = np.clip(raw_for_blc[1::2, 0::2].astype(np.int32) - gb_bl, 0, None).astype(np.uint16)
            b_bl = black_levels_per_channel.get("B", 0)
            corrected[1::2, 1::2] = np.clip(raw_for_blc[1::2, 1::2].astype(np.int32) - b_bl, 0, None).astype(np.uint16)
        elif pattern == "bggr":
            print("DEBUG: BLC: Applying BGGR pattern correction.")
            b_bl = black_levels_per_channel.get("B", 0)
            corrected[0::2, 0::2] = np.clip(raw_for_blc[0::2, 0::2].astype(np.int32) - b_bl, 0, None).astype(np.uint16)
            gb_bl = black_levels_per_channel.get("Gb", 0)
            corrected[0::2, 1::2] = np.clip(raw_for_blc[0::2, 1::2].astype(np.int32) - gb_bl, 0, None).astype(np.uint16)
            gr_bl = black_levels_per_channel.get("Gr", 0)
            corrected[1::2, 0::2] = np.clip(raw_for_blc[1::2, 0::2].astype(np.int32) - gr_bl, 0, None).astype(np.uint16)
            r_bl = black_levels_per_channel.get("R", 0)
            corrected[1::2, 1::2] = np.clip(raw_for_blc[1::2, 1::2].astype(np.int32) - r_bl, 0, None).astype(np.uint16)
        else:
            print(f"WARNING: BLC: Unsupported Bayer pattern '{pattern}'. Falling back to generic black level.")
            black_level = config.get("black_level", 64)
            corrected = np.clip(raw_for_blc.astype(np.int32) - black_level, 0, None).astype(np.uint16)
    else:
        black_level = config.get("black_level", 64) 
        print(f"DEBUG: BLC: Applying generic black level '{black_level}'.")
        corrected = np.clip(raw_for_blc.astype(np.int32) - black_level, 0, None).astype(np.uint16)

    # --- 调试可视化（如果需要）---
    # if debug_path:
    #     save_image_debug(corrected, debug_path.replace(".png", "_blc_after.png"), 
    #                      scale=True, max_val_for_scale=max_sensor_value)
    #     print(f"DEBUG: BLC: Saved image after BLC to {debug_path.replace('.png', '_blc_after.png')}")

    return corrected
