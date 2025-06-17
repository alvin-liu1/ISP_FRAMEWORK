# pipeline.py
# ---------------------
# ISP 主流程控制器（Pipeline）
# ✅ 新手友好版，逐步执行每个模块，每步保存图像

import yaml
import os
import numpy as np 

from raw_loader.raw_reader import read_raw
# 导入所有 ISP 阶段模块，包括新增的 denoise 和 sharpen
from stages import blc, lsc, wb, ccm, demosaic, denoise, sharpen, gamma, tonemapping 
from utils.image_io import save_image_debug, save_image

class ISPPipeline:
    def __init__(self, config_file):
        with open(config_file, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # LSC 增益图的预计算（如果需要，目前在 LSC 模块内部处理）
        self.lsc_gain_map = None # 保持此行，作为未来优化的占位符
        
    def run(self):
        cfg = self.config
        print(f"开始处理：{cfg['raw']['path']}")
        
        # 读取原始 RAW 数据
        raw = read_raw(cfg['raw'])
        print(f"图像尺寸：{raw.shape}")
        
        # 调试：打印原始 RAW 数据范围和类型，这在排查早期问题时很有用
        print(f"DEBUG: 原始 RAW 数据类型: {raw.dtype}, 最小值: {raw.min()}, 最大值: {raw.max()}")

        # 创建调试图像保存目录
        debug_dir = cfg['output'].get('debug_dir', 'output')
        os.makedirs(debug_dir, exist_ok=True)

        # Step 1: 黑电平校正 (Black Level Correction - BLC)
        # 移除图像中的暗电流和传感器固有的零点偏差
        if cfg['blc']['enable']:
            raw = blc.apply(raw, cfg['blc'])
            # BLC 后仍然是 RAW 数据，通常是 uint16 或浮点，保存时需要缩放以便查看
            save_image_debug(raw, os.path.join(debug_dir, 'step1_blc.png'), scale=True) 

        # Step 2: 镜头阴影校正 (Lens Shading Correction - LSC)
        # 补偿镜头造成的图像边缘变暗现象
        if cfg['lsc']['enable']:
            raw = lsc.apply(raw, cfg['lsc'])
            # LSC 后仍然是 RAW 数据，可能值域略有变化，保存时需要缩放
            save_image_debug(raw, os.path.join(debug_dir, 'step2_lsc.png'), scale=True)

        # Step 3: 去马赛克 (Demosaic)
        # 将 Bayer 模式的 RAW 数据转换为完整的 RGB 图像
        # 这一步之后，数据会从 RAW 格式转换为 RGB 浮点格式 (0-1 范围)
        if cfg['demosaic']['enable']:
            rgb = demosaic.apply(raw, cfg['demosaic'])
            # 去马赛克后是 0-1 范围的 RGB 浮点图像，保存时可以不缩放查看实际亮度
            save_image_debug(rgb, os.path.join(debug_dir, 'step3_demosaic.png'), scale=False)
        else:
            raise Exception("去马赛克（Demosaic）模块必须启用才能获得 RGB 图像。")

        # Step 4: 白平衡 (White Balance - WB)
        # 校正图像在不同色温光源下的偏色，使白色物体在图像中显示为白色
        if cfg['wb']['enable']:
            rgb = wb.apply(rgb, cfg['wb'])
            print(f"→ WB 输出最大值：{rgb.max():.4f}")
            print(f"→ WB 输出最小值：{rgb.min():.4f}")
            # 调试：打印图像中心像素值，有助于了解颜色平衡情况
            # 注意：如果图像尺寸不是 1920x1080，这里的 (540, 960) 需要调整
            print(f"→ WB 输出中心像素值 (R, G, B)：{rgb[rgb.shape[0]//2, rgb.shape[1]//2]}") 
            # 白平衡后仍是 0-1 范围的 RGB 浮点图像
            save_image_debug(rgb, os.path.join(debug_dir, 'step4_wb.png'), scale=False)

        # --- 新增模块 ---

        # Step 5: 去噪 (Denoise)
        # 减少图像传感器产生的随机噪声，使图像更平滑
        # 通常放在 WB 之后，CCM 之前
        if cfg['denoise']['enable']:
            print("DEBUG: Denoise configuration being used:")
            # 打印整个 denoise 配置字典
            print(cfg['denoise']) 
            rgb = denoise.apply(rgb, cfg['denoise'])
            print(f"→ 去噪 输出最大值：{rgb.max():.4f}")
            print(f"→ 去噪 输出最小值：{rgb.min():.4f}")
            # 去噪后仍是 0-1 范围的 RGB 浮点图像
            save_image_debug(rgb, os.path.join(debug_dir, 'step5_denoise.png'), scale=False)

        # --- 原来的 Step 5 变为 Step 6 ---

        # Step 6: 颜色校正矩阵 (Color Correction Matrix - CCM)
        # 校正图像在标准色彩空间下的颜色偏差，使颜色更准确
        if cfg['ccm']['enable']:
            rgb = ccm.apply(rgb, cfg['ccm'])
            print(f"→ CCM 输出最大值：{rgb.max():.4f}")
            print(f"→ CCM 输出最小值：{rgb.min():.4f}")
            # CCM 后仍是 0-1 范围的 RGB 浮点图像
            save_image_debug(rgb, os.path.join(debug_dir, 'step6_ccm.png'), scale=False)

        # --- 原来的 Step 6 变为 Step 7 ---

        # Step 7: 伽马校正 (Gamma Correction)
        # 调整图像亮度，使其在显示器上看起来更自然，符合人眼的感知
        if cfg['gamma']['enable']:
            rgb = gamma.apply(rgb, cfg['gamma'])
            print(f"→ 伽马校正 输出最大值：{rgb.max():.4f}")
            print(f"→ 伽马校正 输出最小值：{rgb.min():.4f}")
            # 伽马校正后是 0-1 范围的 RGB 浮点图像（非线性），保存时通常不缩放
            save_image_debug(rgb, os.path.join(debug_dir, 'step7_gamma.png'), scale=False)


        if cfg.get('tonemapping', {}).get('enable', False):
            rgb = tonemapping.apply(rgb, cfg['tonemapping'])
            save_image_debug(rgb, os.path.join(debug_dir, 'step8_tonemapping.png'))
        # --- 新增模块 ---

        # Step 8: 锐化 (Sharpen)
        # 增强图像中的边缘和细节，使图像更清晰
        # 通常放在伽马校正之后
        if cfg['sharpen']['enable']:
            rgb = sharpen.apply(rgb, cfg['sharpen'])
            print(f"→ 锐化 输出最大值：{rgb.max():.4f}")
            print(f"→ 锐化 输出最小值：{rgb.min():.4f}")
            # 锐化后仍是 0-1 范围的 RGB 浮点图像
            save_image_debug(rgb, os.path.join(debug_dir, 'step9_sharpen.png'), scale=False)


        # --- 新增模块：抖动 (Dithering) ---
        # 抖动用于缓解将浮点数图像转换为 8 位整数时可能出现的量化伪影（断层）
        # 它通过在像素值中引入微小的随机噪声，使颜色过渡看起来更平滑。
        
        # 从配置中获取抖动强度。通常 dither_strength 设置为 0.5 到 1.0。
        # 例如，0.5 意味着引入 +/- 0.5 个 8位像素值的噪声。
        dither_strength = cfg['output'].get('dither_strength', 0.5) 
        
        # 仅当 dither_strength 大于 0 时才应用抖动
        if dither_strength > 0:
            # 生成与 RGB 图像相同形状的随机噪声。
            # np.random.rand() 生成 0 到 1 之间的均匀随机数。
            # (np.random.rand(*rgb.shape) - 0.5) 将范围调整为 -0.5 到 0.5。
            # 乘以 dither_strength 是为了控制抖动强度。
            # 乘以 (1.0 / 255.0) 是因为 rgb 已经是 0-1 范围，我们需要将抖动噪声也调整到这个比例。
            # 例如，dither_strength=0.5 意味着噪声范围是 +/- 0.5 / 255.0 在 0-1 浮点数比例中。
            noise = (np.random.rand(*rgb.shape).astype(np.float32) - 0.5) * dither_strength * (1.0 / 255.0)
            
            # 将噪声添加到 RGB 图像中
            rgb_dithered = rgb + noise
            print(f"→ 应用抖动，强度：{dither_strength}")
        else:
            rgb_dithered = rgb # 如果不启用抖动，则使用原始 rgb


        # 最终保存图像
        # 将最终的 0-1 范围浮点图像缩放到 0-255 并转换为 uint8 类型，以便保存为标准图片格式
        final_output_for_save = (rgb * 255).astype(np.uint8)
        save_image(final_output_for_save, cfg['output']['path'])
        print(f"✅ 处理完成，输出已保存至：{cfg['output']['path']}")