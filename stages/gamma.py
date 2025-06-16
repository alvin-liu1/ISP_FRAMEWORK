# ---------------------
# Gamma 校正模块
# ✅ 通常放在 ISP 流程末尾，使图像对比更符合人眼感知。

# stages/gamma.py
# ---------------------
# Gamma 校正模块
# ✅ 通常放在 ISP 流程末尾，使图像对比更符合人眼感知。

import numpy as np

def apply(rgb, config):
    gamma_value = config.get("value", 2.2) # 从配置中获取伽马值，默认 2.2

    # --- 关键：确保输入的 rgb 已经是 0-1 范围的 float32 ---
    # 这一步是防护性的裁剪和类型转换。
    # 假设来自 CCM 模块的 rgb 已经是 0-1 范围的 float32。
    # 如果前一个模块严格遵守 0-1 范围输出，那么这里 np.clip 主要是处理浮点运算误差。
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    # --- 检查 gamma_value 的有效性 ---
    # 防止因配置错误导致数学运算异常
    if not isinstance(gamma_value, (int, float)) or gamma_value <= 0:
        print(f"警告: gamma.py 接收到无效的伽马值 '{gamma_value}'，将使用默认值 2.2。")
        gamma_value = 2.2
    
    # --- 核心伽马校正：对 0-1 范围的线性数据应用幂函数 ---
    # 公式是 output = input^(1/gamma_value)
    # 这一步将线性亮度转换为非线性亮度，使其更符合人眼感知，并使图像看起来更亮。
    try:
        power_exponent = 1.0 / gamma_value
        corrected = np.power(rgb, power_exponent)
    except ZeroDivisionError:
        print("错误: 伽马值设置为 0，无法进行伽马校正。返回原始图像。")
        corrected = rgb # 如果伽马值为 0，无法计算，返回原始 rgb
    except Exception as e:
        print(f"伽马校正中发生未知错误: {e}。返回原始图像。")
        corrected = rgb # 捕获其他潜在异常

    # --- 裁剪输出回 0-1 范围 ---
    # 伽马校正后的值理论上也在 0-1 之间，但浮点运算可能导致微小偏差。
    corrected = np.clip(corrected, 0, 1)

    return corrected # 返回 0-1 范围的浮点数图像 (非线性亮度)