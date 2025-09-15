import numpy as np

def apply(rgb, config):
    """色彩空间转换和色域管理"""
    input_space = config.get("input_space", "sRGB")
    output_space = config.get("output_space", "sRGB")
    gamut_mapping = config.get("gamut_mapping", "clip")
    
    if input_space == output_space:
        return rgb
    
    # sRGB to Rec.2020 转换矩阵
    if input_space == "sRGB" and output_space == "rec2020":
        transform_matrix = np.array([
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956]
        ])
    else:
        # 默认单位矩阵
        transform_matrix = np.eye(3)
    
    # 应用色彩空间转换
    rgb_flat = rgb.reshape(-1, 3)
    rgb_transformed = np.dot(rgb_flat, transform_matrix.T)
    rgb_out = rgb_transformed.reshape(rgb.shape)
    
    # 色域映射
    if gamut_mapping == "clip":
        rgb_out = np.clip(rgb_out, 0, 1)
    elif gamut_mapping == "compress":
        # 软压缩超出色域的颜色
        rgb_out = rgb_out / (1 + rgb_out)
    
    return rgb_out.astype(np.float32)