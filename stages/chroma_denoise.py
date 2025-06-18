
import numpy as np

def apply(rgb, config):
    """
    Chroma Denoise 模块
    - rgb: 输入 RGB 图像，要求 0~1 float32
    - config: 配置参数
    """
    luma_threshold = config.get("luma_threshold", 0.2)
    chroma_strength = config.get("chroma_strength", 0.7)

    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    gray = np.mean(rgb, axis=2, keepdims=True) # (H, W, 1)

    chroma = rgb - gray # (H, W, 3) - (H, W, 1) -> 广播后 (H, W, 3)

    mask = gray < luma_threshold # (H, W, 1)

    # --- 修复方案 ---
    # 问题在于 `rgb[mask]`，`gray[mask]`，`chroma[mask]` 这三个部分的索引
    # 它们期望 `mask` 能与它们自身进行广播，或者 `mask` 的维度能完全匹配。
    # 你的 `mask` 是 (H, W, 1)，而 `rgb` 和 `chroma` 是 (H, W, 3)。
    # NumPy 在布尔索引赋值时，要求索引数组能够精确地匹配或广播到被索引数组的形状。
    # 当 `mask` 试图索引 `rgb` 的第三个轴（大小为3）时，`mask` 在该轴上的大小为1，
    # 导致了维度不匹配。

    # 最直接的解决办法是，确保 `mask` 的最后一个维度也能和 `rgb` 匹配。
    # 但更常见且简洁的 NumPy 习惯是利用 `mask` 的扁平化版本。

    # 尝试将 `mask` 扁平化，然后让 NumPy 自动广播到所有通道
    # 这将 `(H, W, 1)` 的 mask 变成 `(H, W)`，然后应用到 `(H, W, 3)`
    # 这样做，NumPy 会将 (H, W) 的 mask 应用到 (H, W, :) 的所有通道。
    
    # 重点：确保操作发生在正确的形状上。
    # 当 `mask` 是 `(H, W, 1)` 并且应用于 `rgb` (或其他 `(H, W, C)` 数组) 时
    # 报错是因为 `rgb[mask]` 的布尔索引期望 `mask` 的最后一个维度与 `rgb` 的最后一个维度匹配，
    # 或者 `mask` 本身就是一个能够完全决定哪些元素被选中的布尔数组。

    # 解决方法：让 `mask` 扁平化，或者明确地复制 `mask` 到所有通道。
    # 如果 `mask` 是 `(H, W, 1)`，它只能有效索引 `(H, W, 1)` 的数组。
    # 对于 `(H, W, 3)` 的数组，你需要一个 `(H, W, 3)` 的 `mask` 或者 `(H, W)` 的 `mask`
    # 才能让它在所有通道上生效。

    # 最可靠的修复方式是：
    # 1. 计算出需要修改的 `mask_indices`，它们是 `(H, W)` 形状的布尔值
    mask_indices = mask.squeeze(axis=2) # 形状变为 (H, W)

    # 2. 对 `rgb`、`gray` 和 `chroma` 应用这个 `mask_indices`。
    # 当 `mask_indices` 是 `(H, W)` 形状时，它能正确地广播到 `(H, W, C)` 数组的所有通道。
    rgb[mask_indices] = gray[mask_indices] + chroma[mask_indices] * (1 - chroma_strength)

    return np.clip(rgb, 0, 1)