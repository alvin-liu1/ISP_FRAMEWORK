# image_io.py
import cv2
import numpy as np

def save_image_debug(img, path, scale=False, reference_max=None):
    img_float = img.astype(np.float32)

    if scale:
        if reference_max is not None:
            max_val = reference_max
        else:
            max_val = img_float.max()
            
        if max_val > 0:
            img_scaled = img_float / max_val * 255.0
            # 关键修复：防止uint8溢出
            img_processed = np.clip(img_scaled, 0, 255).astype(np.uint8)
        else:
            img_processed = np.zeros_like(img_float, dtype=np.uint8)
    else:
        img_processed = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
            
    cv2.imwrite(path, img_processed)

# save_image 函数保持不变，因为它已经按照期望工作
def save_image(img, path):
    #img = np.clip(img, 0, 1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    
    
    
