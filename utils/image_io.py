# image_io.py
import cv2
import numpy as np

def save_image_debug(img, path, scale=False):
    # 确保 img 是浮点类型以进行精确计算
    img_float = img.astype(np.float32)

    if scale:
        # 动态拉伸逻辑 (保持不变)
        img_processed = np.clip(img_float, 0, None) # 裁剪负值
        max_val = img_processed.max()         # 找到当前图像的最大值
        if max_val > 0:
            img_processed = img_processed / max_val * 255.0 # 将最大值拉伸到 255
        else:
            img_processed = img_processed * 0
        img_processed = img_processed.astype(np.uint8)
    else:
        # === 关键修正：简化并确保 0-1 浮点数正确缩放到 0-255 uint8 ===
        # 无条件地将 0-1 浮点数乘以 255 并转换为 uint8
        # 因为我们知道在这个模式下，img 应该就是 0-1 范围的浮点数
        img_processed = (img_float * 255.0)
        img_processed = np.clip(img_processed, 0, 255).astype(np.uint8)
            
    cv2.imwrite(path, img_processed)

# save_image 函数保持不变，因为它已经按照期望工作
def save_image(img, path):
    #img = np.clip(img, 0, 1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)