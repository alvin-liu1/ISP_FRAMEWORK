
# 文件：test/test_blc.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from stages import blc
from utils.image_io import save_image_debug

def normalize_for_display(img):
    img = np.clip(img, 0, None)
    img = img.astype(np.float32)
    img = img / img.max() * 255.0
    return img.astype(np.uint8)

if __name__ == "__main__":
    # 模拟 raw 数据
    dummy_raw = np.random.randint(64, 300, size=(100, 100)).astype(np.float32)
    config = {"black_level": 64}

    # 原图
    raw_vis = normalize_for_display(dummy_raw)
    cv2.imshow("Before BLC", raw_vis)

    # 执行 BLC
    result = blc.apply(dummy_raw, config)

    # 处理后图
    blc_vis = normalize_for_display(result)
    cv2.imshow("After BLC", blc_vis)

    # 可选：保存调试图
    save_image_debug(dummy_raw, "output/test_blc_before.png", scale=True)
    save_image_debug(result, "output/test_blc_after.png", scale=True)

    print("✅ 显示中，按任意键退出窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
