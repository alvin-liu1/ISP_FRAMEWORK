# --------------------- 
# 读取 RAW 图像（10bit, unpacked 格式） 
# 返回 numpy 格式，供后续处理 
# 新手建议：确保 raw 的路径和分辨率与 config.yaml 一致


import numpy as np 
import os 
def read_raw(cfg):     
     path = cfg['path']     
     width = cfg['width']     
     height = cfg['height']     
     bit_depth = cfg['sensor_bit_depth'] 
     if bit_depth == 10:         
         # 10bit, 每像素2字节对齐（常见 unpacked 格式）         
         raw = np.fromfile(path, dtype=np.uint16)         
         raw = raw.reshape((height, width))     
     else:         
         raise NotImplementedError("Only 10bit raw supported")
      
     return raw.astype(np.float32)
