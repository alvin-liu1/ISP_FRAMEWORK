import numpy as np

def apply(raw, config):
    h, w = raw.shape
    cx, cy = config.get("center", [w//2, h//2])
    radius = config.get("radius", min(h, w) // 2 - 10)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist <= radius

    raw_masked = raw.copy()
    raw_masked[~mask] = 0
    return raw_masked
