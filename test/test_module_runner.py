import numpy as np
import os
from utils.image_io import save_image_debug

# 可测的模块列表
from stages import blc, lsc, wb, ccm, gamma, demosaic

# 测试函数集合
def test_module(module_name):
    os.makedirs("output", exist_ok=True)

    if module_name == "blc":
        dummy = np.random.randint(64, 400, size=(100, 100)).astype(np.float32)
        out = blc.apply(dummy, {"black_level": 64})
        save_image_debug(out, "output/test_blc.png", scale=True)

    elif module_name == "lsc":
        img = np.ones((100, 100), dtype=np.float32) * 100
        img[50, 50] = 150  # 中心亮
        out = lsc.apply(img, {})
        save_image_debug(out, "output/test_lsc.png", scale=True)

    elif module_name == "demosaic":
        # 模拟 Bayer 格式 raw 图（RGGB）随机图
        raw = np.tile(np.array([[64, 32], [32, 128]]), (50, 50)).astype(np.float32)
        out = demosaic.apply(raw, {"bayer_pattern": "rggb"})
        save_image_debug(out, "output/test_demosaic.png", scale=True)

    elif module_name == "wb":
        img = np.ones((100, 100, 3), dtype=np.float32) * [40, 80, 160]
        out = wb.apply(img, {"gains": [2.0, 1.0, 1.0]})
        save_image_debug(out, "output/test_wb.png", scale=True)

    elif module_name == "ccm":
        img = np.ones((100, 100, 3), dtype=np.float32) * [50, 100, 200]
        matrix = [[1.5, -0.3, -0.2],
                  [-0.2, 1.3, -0.1],
                  [0.0, -0.6, 1.6]]
        out = ccm.apply(img, {"matrix": matrix})
        save_image_debug(out, "output/test_ccm.png", scale=True)

    elif module_name == "gamma":
        img = np.ones((100, 100, 3), dtype=np.float32) * 128
        out = gamma.apply(img, {"value": 2.2})
        save_image_debug(out, "output/test_gamma.png", scale=True)

    else:
        print(f"❌ 模块 {module_name} 未支持")

# 示例入口（手动指定测试）
if __name__ == "__main__":
    for name in ["blc", "lsc", "wb", "ccm", "gamma", "demosaic"]:
        print(f"✅ 正在测试模块：{name}")
        test_module(name)
