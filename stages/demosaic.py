# ---------------------
# 去马赛克（Demosaic）模块
# ✅ 通常在 BLC 之后、WB/CCM 之前执行。

# stages/demosaic.py
import cv2
import numpy as np

def apply(raw, config):
    method = config.get("method", "opencv")
    pattern = config.get("bayer_pattern", "rggb")
    
    # 获取位深配置
    bit_depth_cfg = config.get('bit_depth_management', {})
    input_bit_depth = bit_depth_cfg.get('lsc_output', 12)
    
    print(f"Demosaic: 输入{input_bit_depth}bit → Float32 HDR")
    print(f"Demosaic: 输入范围 [{raw.min():.1f}, {raw.max():.1f}]")
    
    # 方案：转换为16bit进行demosaic，然后归一化为float32
    max_input_value = (2**input_bit_depth) - 1  # 4095 for 12-bit
    
    # 将12bit数据缩放到16bit范围 (0-65535)
    raw_16bit = (raw / max_input_value * 65535.0).astype(np.uint16)
    
    # 选择demosaic算法
    if method == "opencv_ea":
        code = cv2.COLOR_BayerRG2RGB_EA
    else:
        code = {
            "rggb": cv2.COLOR_BayerRG2RGB_VNG,
            "bggr": cv2.COLOR_BayerBG2RGB_VNG,
            "grbg": cv2.COLOR_BayerGR2RGB_VNG,
            "gbrg": cv2.COLOR_BayerGB2RGB_VNG,
        }.get(pattern.lower(), cv2.COLOR_BayerRG2RGB_VNG)
    
    # OpenCV demosaic处理（16bit）
    rgb_16bit = cv2.cvtColor(raw_16bit, code)
    
    # 转换回float32并归一化到0-1范围
    rgb = rgb_16bit.astype(np.float32) / 65535.0
    
    print(f"Demosaic: 输出Float32 HDR范围 [{rgb.min():.4f}, {rgb.max():.4f}]")
    
    return rgb
    
 
