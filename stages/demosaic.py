# ---------------------
# 去马赛克（Demosaic）模块
# ✅ 通常在 BLC 之后、WB/CCM 之前执行。

"""import cv2
import numpy as np

def apply(raw, config):
    method = config.get("method", "opencv")
    pattern = config.get("bayer_pattern", "rggb")
    code = {
        "rggb": cv2.COLOR_BayerRG2BGR,
        "bggr": cv2.COLOR_BayerBG2BGR,
        "grbg": cv2.COLOR_BayerGR2BGR,
        "gbrg": cv2.COLOR_BayerGB2BGR,
    }.get(pattern.lower(), cv2.COLOR_BayerRG2BGR)

    raw_uint16 = np.clip(raw, 0, 1023).astype(np.uint16)
    raw_8bit = (raw_uint16 / 4).astype(np.uint8)
    rgb = cv2.cvtColor(raw_8bit, code)
    return rgb.astype(np.float32)"""
    
    
# stages/demosaic.py
import cv2
import numpy as np

def apply(raw, config):
    method = config.get("method", "opencv")
    pattern = config.get("bayer_pattern", "rggb")
    # 推荐使用 VNG (Variable Number of Gradients) 算法，效果通常比简单的 BayerRG2BGR_COLOR_BGR 好
    code = {
        "rggb": cv2.COLOR_BayerRG2BGR_VNG,
        "bggr": cv2.COLOR_BayerBG2BGR_VNG,
        "grbg": cv2.COLOR_BayerGR2BGR_VNG,
        "gbrg": cv2.COLOR_BayerGB2BGR_VNG,
    }.get(pattern.lower(), cv2.COLOR_BayerRG2BGR_VNG)

    # 从配置中获取原始传感器的位深。例如 10 代表 10-bit RAW。
    sensor_bit_depth = config.get("sensor_bit_depth", 10)
    
    # 原始 RAW 数据的最大理论值（例如 10-bit 是 1023，12-bit 是 4095）
    original_raw_max_value = (2**sensor_bit_depth) - 1

    # raw_to_16bit_scale_factor：将原始 RAW 整数值映射到 0-65535 uint16 范围。
    raw_to_16bit_scale_factor = config.get("raw_to_16bit_scale_factor", 64) # 默认 64 适用于 10-bit 转 16-bit

    # --- 关键修正：为了适应 OpenCV 的 CV_8U 要求 ---

    # 1. 将 `raw` 输入（假设是原始位深的整数值，或者已左移的 uint16）
    # 首先，将其映射到 0-65535 的 uint16 范围，这是我们之前做过的，确保数据密度。
    raw_scaled_to_16bit_uint16 = np.clip(raw * raw_to_16bit_scale_factor, 0, 65535).astype(np.uint16)

    # 2. **新增步骤：** 将 0-65535 范围的 uint16 数据**临时转换**为 0-255 的 uint8。
    # 这一步是为了满足 OpenCV demosaicing 函数的 `CV_8U` 要求。
    # 我们将 uint16 范围除以 257 (约等于 65535 / 255)，将其缩放到 0-255 范围。
    # 257 优于 256 是为了在 16bit 到 8bit 映射时更均匀，因为 65535 / 255 = 257。
    raw_for_opencv_8bit = (raw_scaled_to_16bit_uint16 // 257).astype(np.uint8) # 使用整数除法，并转换为 uint8

    # 3. 核心去马赛克：现在传入 8-bit 数据给 OpenCV
    # cv2.cvtColor(uint8_bayer, ...) 会输出 0-255 的 uint8 BGR 图像。
    rgb_demosaiced_8bit = cv2.cvtColor(raw_for_opencv_8bit, code)

    # 4. **新增步骤：** 将 0-255 范围的 uint8 BGR 图像转换回 0-1 范围的 float32。
    # 因为我们的 ISP 流程后续步骤都期望 0-1 float32。
    # 这里我们直接除以 255.0，将其从 8 位范围归一化到 0-1 范围。
    rgb_float_final = rgb_demosaiced_8bit.astype(np.float32) / 255.0
    
    # 5. 最终裁剪到 0-1 范围，防止微小的浮点运算误差。
    return np.clip(rgb_float_final, 0, 1)