
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
    denoise,chroma_denoise, sharpen, gamma, tonemapping, super_resolution, \
    noise_estimation, color_space, dpc
from utils.image_io import save_image_debug, save_image

def log_data_range(rgb, step_name):
    """监控数据范围，帮助调试"""
    print(f"→ {step_name}: 范围[{rgb.min():.3f}, {rgb.max():.3f}], "
          f"均值{rgb.mean():.3f}, "
          f"99%分位数{np.percentile(rgb, 99):.3f}")

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

            # --- ISP 流程开始 ---
            
            # Step 0: 坏点校正 (Dead Pixel Correction - DPC) - 在BLC之前处理
            if cfg.get('dpc', {}).get('enable', False):
                dpc_cfg = cfg['dpc'].copy()
                dpc_cfg['bayer_pattern'] = cfg['demosaic'].get('bayer_pattern', 'rggb')
                raw = dpc.apply(raw, dpc_cfg)
                save_image_debug(raw, os.path.join(current_debug_dir, 'step0_dpc.png'), scale=True)
            
            # Step 1: 鱼眼掩膜 (Fisheye Mask) - 可选预处理
            if cfg.get('fisheye_mask', {}).get('enable', False):
                raw = fisheye_mask.apply(raw, cfg['fisheye_mask'])
                save_image_debug(raw, os.path.join(current_debug_dir, 'step1_fisheye_mask.png'), scale=True)

            # 确保所有模块都能访问位深配置
            bit_depth_cfg = cfg.get('bit_depth_management', {})
            
            # Step 2: 黑电平校正 (Black Level Correction - BLC)
            if cfg['blc']['enable']:
                blc_cfg = cfg['blc'].copy()
                blc_cfg['bit_depth_management'] = bit_depth_cfg
                raw = blc.apply(raw, blc_cfg)
                blc_max = raw.max()  # 记录BLC后的最大值
                save_image_debug(raw, os.path.join(current_debug_dir, 'step2_blc.png'), scale=True)
            
            # Step 3: 暗部 Clip 保底去噪 (可选)
            if cfg.get('denoise_clip', {}).get('enable', False):
                raw = denoise_clip.apply(raw, cfg['denoise_clip'])
                save_image_debug(raw, os.path.join(current_debug_dir, 'step3_denoise_clip.png'), scale=True) 
                
            # Step 4: 镜头阴影校正 (Lens Shading Correction - LSC)
            if cfg['lsc']['enable']:
                lsc_cfg = cfg['lsc'].copy()
                lsc_cfg['sensor_bit_depth'] = cfg['raw'].get('sensor_bit_depth', 10)
                lsc_cfg['bit_depth_management'] = bit_depth_cfg
                raw = lsc.apply(raw, lsc_cfg)
                # 不使用reference_max，让LSC图像用自己的最大值缩放
                save_image_debug(raw, os.path.join(current_debug_dir, 'step4_lsc.png'), scale=True)

            # Step 5: 噪声估计 (Noise Estimation) - 为后续自适应处理提供信息
            if cfg.get('noise_estimation', {}).get('enable', False):
                raw = noise_estimation.apply(raw, cfg)
                save_image_debug(raw, os.path.join(current_debug_dir, 'step5_noise_estimation.png'), scale=True)
                print(f"→ 噪声估计完成")

            # Step 6: 去马赛克 (Demosaic) - 从RAW转换为RGB
            if cfg['demosaic']['enable']:
                demosaic_cfg = cfg['demosaic'].copy()
                demosaic_cfg['bit_depth_management'] = bit_depth_cfg
                if demosaic_cfg.get('method') == 'rawpy' or demosaic_cfg.get('method') == 'auto':
                    demosaic_cfg['raw_file_path'] = raw_file_path
                
                rgb = demosaic.apply(raw, demosaic_cfg)
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step6_demosaic.png'), scale=False)
            else:
                raise Exception("去马赛克（Demosaic）模块必须启用才能获得 RGB 图像。")

            # Step 7: 色彩空间转换 (Color Space Conversion) - 可选
            if cfg.get('color_space_conversion', {}).get('enable', False):
                rgb = color_space.apply(rgb, cfg['color_space_conversion'])
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step7_color_space.png'), scale=False)

            # Step 8: 白平衡 (保持HDR范围)
            if cfg['wb']['enable']:
                rgb = wb.apply(rgb, cfg['wb'])
                print(f"→ WB 输出范围：{rgb.min():.3f} - {rgb.max():.3f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step8_wb.png'), scale=False)

            # Step 9: 去噪 (在线性HDR空间)
            if cfg['denoise']['enable']:
                rgb = denoise.apply(rgb, cfg['denoise'])
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step9_denoise.png'), scale=False)

            # Step 10: CCM (保持HDR范围)
            if cfg['ccm']['enable']:
                rgb = ccm.apply(rgb, cfg['ccm'])
                print(f"→ CCM 输出范围：{rgb.min():.3f} - {rgb.max():.3f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step10_ccm.png'), scale=False)

            # Step 11: Tone Mapping (HDR → LDR，线性空间)
            if cfg.get('tonemapping', {}).get('enable', False):
                rgb = tonemapping.apply(rgb, cfg['tonemapping'])
                print(f"→ Tone Mapping 输出范围：{rgb.min():.3f} - {rgb.max():.3f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step11_tonemapping.png'), scale=False)

            # Step 12: Gamma (线性 → 非线性，0-1范围)
            if cfg['gamma']['enable']:
                rgb = gamma.apply(rgb, cfg['gamma'])
                print(f"→ Gamma 输出范围：{rgb.min():.3f} - {rgb.max():.3f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step12_gamma.png'), scale=False)

            # Step 13: 色度去噪 (Chroma Denoising) - 针对色彩噪声
            if cfg.get('chroma_denoise', {}).get('enable', False):
                rgb = chroma_denoise.apply(rgb, cfg['chroma_denoise'])
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step13_chroma_denoise.png'), scale=False)

            # Step 14: 锐化 (Sharpen)
            if cfg['sharpen']['enable']:
                rgb = sharpen.apply(rgb, cfg['sharpen'])
                print(f"→ 锐化 输出最大值：{rgb.max():.4f}")
                print(f"→ 锐化 输出最小值：{rgb.min():.4f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step14_sharpen.png'), scale=False)
            
            # Step 15: 超分辨率 (Super Resolution)
            if 'super_resolution' in cfg and cfg['super_resolution'].get('enable', False):
                rgb = super_resolution.apply(rgb, cfg['super_resolution'])
                print(f"→ 超分辨 输出尺寸：{rgb.shape}")
                print(f"→ 超分辨 输出最大值：{rgb.max():.4f}")
                print(f"→ 超分辨 输出最小值：{rgb.min():.4f}")
                save_image_debug(rgb, os.path.join(current_debug_dir, 'step15_super_resolution.png'), scale=False)
            
            # Step 16: 抖动 (Dithering) - 最终输出前的处理
            dither_strength = cfg['output'].get('dither_strength', 0.5)
            if dither_strength > 0:
                noise = (np.random.rand(*rgb.shape).astype(np.float32) - 0.5) * dither_strength * (1.0 / 255.0)
                rgb_dithered = rgb + noise
                print(f"→ 应用抖动，强度：{dither_strength}")
                save_image_debug(rgb_dithered, os.path.join(current_debug_dir, 'step16_dithering.png'), scale=False)
            else:
                rgb_dithered = rgb

            # 最终保存图像
            final_output_for_save = (rgb_dithered * 255).astype(np.uint8) # 使用抖动后的图像
            
            # 以原始文件名命名结果图，并保存到 output_dir
            output_path = os.path.join(output_dir, f"{file_name_without_ext}_processed.png")
            save_image(final_output_for_save, output_path)
            print(f"✅ 文件 '{file_name_with_ext}' 处理完成，输出已保存至：{output_path}")

        print("\n--- 所有文件处理完毕 ---")
