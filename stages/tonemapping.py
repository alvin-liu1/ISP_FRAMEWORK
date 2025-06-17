
import numpy as np

def apply(rgb, config):
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    lift = config.get("lift", 0.1)  # 暗部提升
    roll = config.get("roll", 0.7)  # 高光 rolloff 拐点
    compress = config.get("compress", 0.5)  # 高光压缩强度

    # 暗部提升
    rgb_lifted = rgb + lift
    rgb_lifted = np.clip(rgb_lifted, 0, 1)

    # 高光 roll-off 压制
    mask = rgb_lifted > roll
    rgb_lifted[mask] = roll + (rgb_lifted[mask] - roll) * compress

    return np.clip(rgb_lifted, 0, 1)
