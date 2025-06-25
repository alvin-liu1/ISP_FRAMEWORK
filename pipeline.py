



# pipeline.py
# ---------------------
# ISP 主流程控制器（Pipeline）
# ✅ 新手友好版，逐步执行每个模块，每步保存图像

import yaml
import os
import numpy as np
import glob # 新增：用于查找文件

from raw_loader.raw_reader import read_raw
# 导入所有 ISP 阶段模块，包括新增的 denoise 和 sharpen
from stages import fisheye_mask,denoise_clip,blc, lsc, wb, ccm, demosaic, \
    denoise,chroma_denoise, sharpen, gamma, tonemapping, super_resolution
from utils.image_io import save_image_debug, save_image

class ISPPipeline:
    def __init__(self, config_file):
        with open(config_file, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # LSC 增益图的预计算（如果需要，目前在 LSC 模块内部处理）
        self.lsc_gain_map = None # 保持此行，作为未来优化的占位符
        
    def run(self):
        cfg = self.config
        
        # --- 批量处理逻辑开始 ---
        input_dir = cfg['raw'].get('input_dir')
        output_dir = cfg['output'].get('output_dir', 'output/results/') # 默认值
        debug_base_dir = cfg['output'].get('debug_dir', 'output/debug_steps/') # 默认值

        if not input_dir:
            raise ValueError("config.yaml 中 'raw' 部分必须指定 'input_dir' 用于批量处理。")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        # 确保调试目录的基础路径存在，每个文件会在此目录下创建独立的子目录
        os.makedirs(debug_base_dir, exist_ok=True)

        # 查找所有 .raw 文件
        raw_files = glob.glob(os.path.join(input_dir, "*.raw"))
        if not raw_files:
            print(f"警告: 在目录 '{input_dir}' 中未找到任何 .raw 文件。请检查路径和文件后缀。")
            return

        print(f"在 '{input_dir}' 中找到 {len(raw_files)} 个 RAW 文件进行处理。")

        for raw_file_path in raw_files:
            file_name_with_ext = os.path.basename(raw_file_path)
            file_name_without_ext = os.path.splitext(file_name_with_ext)[0]

            print(f"\n--- 开始处理文件: {file_name_with_ext} ---")
            
            # 为当前文件创建独立的调试目录
            current_debug_dir = os.path.join(debug_base_dir, file_name_without_ext)
            os.makedirs(current_debug_dir, exist_ok=True)

            # 更新配置中的 raw 文件路径，确保 read_raw 读取正确的文件
            # 注意：这里我们修改的是 cfg 字典中 raw 模块的 path，但 read_raw 使用的是 cfg['raw'] 整个字典
            # 因此，我们直接修改传入 read_raw 的字典更合适。
            # 或者，read_raw 应该直接接受文件路径作为参数。
            # 假设 read_raw 接受一个字典，且该字典中包含 'path' 键
            current_raw_cfg = cfg['raw'].copy() # 复制一份，避免修改全局配置
            current_raw_cfg['path'] = raw_file_path

            # 读取原始 RAW 数据
            raw = read_raw(current_raw_cfg) # 传入包含当前文件路径的配置
            print(f"图像尺寸：{raw.shape}")
            
            # 调试：打印原始 RAW 数据范围和类型，这在排查早期问题时很有用
            print(f"DEBUG: 原始 RAW 数据类型: {raw.dtype}, 最小值: {raw.min()}, 最大值: {raw.max()}")

            # --- ISP 流程开始（这部分与你原有代码基本相同，只是保存路径变了） ---
            
            # Step 0: Fisheye Mask
            if cfg.get('fisheye_mask', {}).get('enable', False):
                raw = fisheye_mask.apply(raw, cfg['fisheye_mask'])
                save_image_debug(raw, os.path.join(current_debug_dir, 'step0_mask.png'), scale=True)

            # Step 1: 黑电平校正 (Black Level Correction - BLC)
            if cfg['blc']['enable']:
                raw = blc.apply(raw, cfg['blc'])
                save_image_debug(raw, os.path.join(current_debug_dir, 'step1_blc.png'), scale=True) 

            # Step 1.5: 暗部 Clip 保底去噪
            if cfg.get('denoise_clip', {}).get('enable', False):
                raw = denoise_clip.apply(raw, cfg['denoise_clip'])
                save_image_debug(raw, os.path.join(current_debug_dir, 'step1.5_denoise_clip.png'), scale=True) 
                
            # Step 2: 镜头阴影校正 (Lens Shading Correction - LSC)
            if cfg['lsc']['enable']:
                raw = lsc.apply(raw, cfg['lsc'])
                save_image_debug(raw, os.path.join(current_debug_dir, 'step2_lsc.png'), scale=True)

            # Step 3: 去马赛克 (Demosaic)
            if cfg['demosaic']['enable']:
                # 如果 demosaic 模块支持 raw_file_path，这里需要传入原始 raw_file_path
                # 否则，它将依赖于传入的 numpy `raw` 数组
                demosaic_cfg = cfg['demosaic'].copy() # 复制一份配置
                # 如果 demosaic 模块内部使用 rawpy，需要原始文件路径
                # 假设 rawpy 兼容处理在blc/lsc后的数据，或者它会重新读取文件。
                # 最佳实践是 rawpy直接读取未处理的原始文件，然后isp处理。
                # 但根据你之前描述，rawpy似乎被用于demosaic步骤，且需要原始文件路径。
                # 所以，这里将原始文件的路径传递给demoisac配置。
                if demosaic_cfg.get('method') == 'rawpy' or demosaic_cfg.get('method') == 'auto':
                    demosaic_cfg['raw_file_path'] = raw_file_path # 传递原始RAW文件路径
                
                rgb = demosaic.apply(raw, demosaic_cfg) # 传入当前 RAW 数据和 demosaic 配置
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step3_demosaic.png'), scale=False)
            else:
                raise Exception("去马赛克（Demosaic）模块必须启用才能获得 RGB 图像。")

            # Step 4: 白平衡 (White Balance - WB)
            if cfg['wb']['enable']:
                rgb = wb.apply(rgb, cfg['wb'])
                print(f"→ WB 输出最大值：{rgb.max():.4f}")
                print(f"→ WB 输出最小值：{rgb.min():.4f}")
                print(f"→ WB 输出中心像素值 (R, G, B)：{rgb[rgb.shape[0]//2, rgb.shape[1]//2]}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step4_wb.png'), scale=False)

            # Step 5: 去噪 (Denoise)
            if cfg['denoise']['enable']:
                print("DEBUG: Denoise configuration being used:")
                print(cfg['denoise'])
                rgb = denoise.apply(rgb, cfg['denoise'])
                print(f"→ 去噪 输出最大值：{rgb.max():.4f}")
                print(f"→ 去噪 输出最小值：{rgb.min():.4f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step5_denoise.png'), scale=False)

            # Step 6: 颜色校正矩阵 (Color Correction Matrix - CCM)
            if cfg['ccm']['enable']:
                rgb = ccm.apply(rgb, cfg['ccm'])
                print(f"→ CCM 输出最大值：{rgb.max():.4f}")
                print(f"→ CCM 输出最小值：{rgb.min():.4f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step6_ccm.png'), scale=False)

            # Step: Chroma Denoise
            if cfg.get('chroma_denoise', {}).get('enable', False):
                rgb = chroma_denoise.apply(rgb, cfg['chroma_denoise'])
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step7_chroma_denoise.png'))

            # Step 7: 伽马校正 (Gamma Correction)
            if cfg['gamma']['enable']:
                rgb = gamma.apply(rgb, cfg['gamma'])
                print(f"→ 伽马校正 输出最大值：{rgb.max():.4f}")
                print(f"→ 伽马校正 输出最小值：{rgb.min():.4f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step8_gamma.png'), scale=False)

            # Step 8: Tone Mapping
            if cfg.get('tonemapping', {}).get('enable', False):
                rgb = tonemapping.apply(rgb, cfg['tonemapping'])
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step9_tonemapping.png'))

            # Step 9: 锐化 (Sharpen)
            if cfg['sharpen']['enable']:
                rgb = sharpen.apply(rgb, cfg['sharpen'])
                print(f"→ 锐化 输出最大值：{rgb.max():.4f}")
                print(f"→ 锐化 输出最小值：{rgb.min():.4f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step10_sharpen.png'), scale=False)
            
            # === Step 10: 超分辨 (Super Resolution) ===
            if 'super_resolution' in cfg and cfg['super_resolution'].get('enable', False):
                rgb = super_resolution.apply(rgb, cfg['super_resolution'])
                print(f"→ 超分辨 输出尺寸：{rgb.shape}")
                print(f"→ 超分辨 输出最大值：{rgb.max():.4f}")
                print(f"→ 超分辨 输出最小值：{rgb.min():.4f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step11_super_resolution.png'), scale=False)
            
            # --- 新增模块：抖动 (Dithering) ---
            dither_strength = cfg['output'].get('dither_strength', 0.5)
            if dither_strength > 0:
                noise = (np.random.rand(*rgb.shape).astype(np.float32) - 0.5) * dither_strength * (1.0 / 255.0)
                rgb_dithered = rgb + noise
                print(f"→ 应用抖动，强度：{dither_strength}")
            else:
                rgb_dithered = rgb

            # 最终保存图像
            final_output_for_save = (rgb_dithered * 255).astype(np.uint8) # 使用抖动后的图像
            
            # 以原始文件名命名结果图，并保存到 output_dir
            output_path = os.path.join(output_dir, f"{file_name_without_ext}_processed.png")
            save_image(final_output_for_save, output_path)
            print(f"✅ 文件 '{file_name_with_ext}' 处理完成，输出已保存至：{output_path}")

        print("\n--- 所有文件处理完毕 ---")