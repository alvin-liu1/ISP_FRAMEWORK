

# stages/lsc.py
import numpy as np

def apply(raw, config):
    h, w = raw.shape
    
    # 获取配置参数
    # model_type: 可以是 'linear', 'quadratic', 'cosine_fourth' 或 'polynomial'
    model_type = config.get("model_type", "linear") 
    # strength: 控制增益的强度，即补偿的程度
    strength = config.get("strength", 0.5) 
    # normalize_to_1_after_lsc: 是否在 LSC 后将图像归一化到 0-1 范围
    # 这与 ISP 流程中 Demoisac 之后的归一化理念一致
    normalize_to_1_after_lsc = config.get("normalize_to_1_after_lsc", True)
    
    # 假设 raw 此时是 uint16 或 float32，但未归一化到 0-1
    # 记录原始最大值，以便后续归一化
    # 如果 raw 是 0-原始位深最大值 (例如 1023) 的 uint16/float
    # 则 max_raw_value = config.get("original_raw_max_value", 1023)
    # 如果 raw 是左移填充到 uint16 的 (例如 65472)
    # 则 max_raw_value = config.get("max_raw_sensor_value_mapped_to_uint16", 65472)
    # 为了简化，我们假定 raw 的最大值代表其饱和点，并在 LSC 后将其归一化。
    current_max_value = np.max(raw) if np.max(raw) > 0 else 1.0 # 避免除零

    # 计算到中心的归一化距离
    y, x = np.indices((h, w))
    center_y, center_x = h / 2, w / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # 归一化距离到 0-1 范围，0 为中心，1 为最远角
    normalized_distance = distance / np.max(distance) 

    gain_map = np.ones((h, w), dtype=np.float32)

    if model_type == "linear":
        gain_map = 1.0 + strength * normalized_distance
    elif model_type == "quadratic":
        # 抛物线模型，模拟更平滑的过渡
        gain_map = 1.0 + strength * (normalized_distance**2)
    elif model_type == "cosine_fourth":
        # 余弦四次方定律，更接近真实镜头晕影模型
        # k 是一个系数，控制晕影的强度，可以进一步在 config 中设置
        k = config.get("cosine_fourth_k", 0.5) 
        # 注意：这里需要确保 cos(theta) 不会为 0，且 theta 的范围是 [0, pi/2)
        # 简单实现可以是 1.0 / (cos(angle)**4)
        # 但我们用参数化的方式模拟
        gain_map = 1.0 / (1.0 - strength * normalized_distance**2) # 简化近似
        # 或者更精确的：gain_map = 1.0 / (np.cos(k * normalized_distance)**4) 
        # 但需要处理角度和 cos 的定义域问题，为了简单先用近似。
        # 对于简化版，1.0 + strength * normalized_distance**N 已经不错
        
        # 更常用且安全的模拟：
        # 对于晕影校正，通常增益图是 1 / (1 + C1*r^2 + C2*r^4 + ...)
        # 简化为：1 + strength * normalized_distance**4 也可以模拟类似效果，并且更安全。
        gain_map = 1.0 + strength * (normalized_distance**4) 
        
    elif model_type == "polynomial":
        # 多项式模型，更灵活，可以设置多个系数
        # coefficients: [c0, c1, c2, c3, ...] 对应 1 + c1*r + c2*r^2 + ...
        coefficients = config.get("coefficients", [0, 0.2, 0.3]) # 1 + 0.2*r + 0.3*r^2
        gain_map = 1.0 + np.polyval(coefficients[::-1], normalized_distance) # numpy.polyval expects coeffs from highest to lowest power
        # 如果系数是从低到高，可以直接：
        # gain_map = coefficients[0] + coefficients[1]*normalized_distance + coefficients[2]*normalized_distance**2 + ...
        gain_map = np.zeros_like(normalized_distance)
        for i, c in enumerate(coefficients):
            gain_map += c * (normalized_distance**i)
        # 通常 LSC 增益图在中心是 1.0，边缘更大。所以通常是 1.0 + C1*r^2 + C2*r^4
        # 这里的 coefficients 应该是不包含 1.0 的部分
        # 比如 coefficients = [C1, C2]，则 gain_map = 1.0 + C1*r^2 + C2*r^4
        # 为简化，假设 coefficients 是直接用于 1 + poly(r) 的额外增益部分
        # 假设 coefficients 是 [c1, c2]，那么 gain_map = 1 + c1*r^2 + c2*r^4
        if len(coefficients) > 0:
             gain_map = 1.0 + coefficients[0] * normalized_distance
        if len(coefficients) > 1:
             gain_map += coefficients[1] * (normalized_distance**2)
        if len(coefficients) > 2:
             gain_map += coefficients[2] * (normalized_distance**4) # 常用偶数次幂
        # 你可以根据需要扩展多项式阶数。

    else:
        print(f"Warning: Unknown LSC model type '{model_type}'. Using default linear model.")
        gain_map = 1.0 + strength * normalized_distance

    # 应用增益图
    corrected_raw = raw * gain_map
    
    # 关键：LSC 后的归一化
    # 如果设置为 True，我们期望 LSC 后的数据也归一化到 0-1 范围
    if normalize_to_1_after_lsc:
        # 这里需要一个正确的归一化分母。
        # 如果 `raw` 是 0-original_raw_max_value 的 uint16/float
        # 那么 LSC 后的最大值就应该是 original_raw_max_value * (最大增益)
        # 为了保证 1.0 对应饱和，通常将校正后的图像**裁剪**到原始饱和点。
        # demosaic 模块会负责将其归一化到 0-1。
        # 因此，这里的裁剪到 current_max_value 是合理的。
        # 考虑到 demosaic 会进行 0-1 归一化，LSC 只需要负责补偿晕影，
        # 并裁剪到原始数据的有效范围，而不是 0-1。
        # 假设 raw 传入时其最大值是 `max_pixel_value_before_lsc`
        # 那么校正后也应该裁剪到这个值。
        
        # 为了避免 LSC 导致过曝（像素值超过传感器饱和点），通常会裁剪。
        # 但是，我们不应该在这里进行 0-1 归一化，因为 Demosaic 才是归一化的最佳位置。
        # 因此，这里的 `normalize_to_1_after_lsc` 选项应该被移除，
        # 并且 LSC 应该输出与输入 `raw` 相同范围的数据（例如 uint16，或未归一化的浮点）。
        # 我之前的建议是让 demosiac 进行第一次 0-1 归一化。
        
        # 重新考虑数据流：
        # read_raw -> BLC -> LSC -> Demosaic (这里进行 0-1 归一化) -> WB -> CCM -> Gamma
        # 所以 LSC 应该输出与 BLC/read_raw 相同的数值范围。
        # 比如：如果 raw 是 10bit (0-1023) 的 uint16，LSC 也输出 0-1023 (可能更高，需要裁剪)。
        # 如果 raw 是 16bit (0-65535) 的 uint16，LSC 也输出 0-65535。
        
        # 所以，我们应该裁剪到 `current_max_value` 或 `(2**sensor_bit_depth - 1)`
        # 或 `max_raw_sensor_value_mapped_to_uint16` (如果 raw 已经被左移)。
        # 这里使用 np.clip(..., 0, current_max_value) 是一个相对安全的选择，
        # 它确保了 LSC 不会将像素推到超过原始饱和点。
        # 但是，更精确的做法是裁剪到 `config` 中定义的原始传感器的最大值或其 16 位映射值。
        
        # 将裁剪上限设置为 raw 的 dtype 的最大值，或配置中的最大值。
        # 假设 raw 在 BLC/LSC 之前是 uint16，或某个 float 范围。
        # 这里我们假定 raw 经过 BLC 之后，其饱和点仍然是其原始位深对应的最大值。
        # demosaic 的 config 中已经有了 `max_raw_sensor_value_mapped_to_uint16`。
        # 我们可以通过 config 将这个值传递给 LSC。
        lsc_clip_max_value = config.get("max_value_for_clip", current_max_value) 
        # current_max_value 只是当前图像的最大值，不是理论饱和值。
        # 从 config 中获取理论饱和值更准确。
        
        # 更推荐的 LSC 裁剪上限：
        # 如果 raw 是 uint16，则裁剪到 65535
        if raw.dtype == np.uint16:
            lsc_clip_max_value = 65535
        # 如果 raw 是 float32，且我们知道它代表 0-max_raw_value 的范围，
        # 那么就裁剪到那个 max_raw_value。
        # 我们可以从 pipeline 中传递 demosaic 模块使用的 `normalization_denominator` 或
        # `max_raw_sensor_value_mapped_to_uint16` 给 LSC 作为裁剪上限。
        
        # 为了简化，并且与 demosaic 的输入保持一致，我们假定 raw 保持其原始类型和范围。
        # LSC 只对亮度进行补偿，不改变其数据类型或归一化。
        # 因此，这里仅仅裁剪负值，确保不会出现负数，同时避免了不必要的归一化。
        return np.clip(corrected_raw, 0, None) # 仅裁剪负值，保持原始范围
    
    # 返回与输入 raw 相同类型的数据（通常是 uint16 或 float32，未归一化到 0-1）
    return corrected_raw.astype(raw.dtype) if raw.dtype == np.uint16 else corrected_raw