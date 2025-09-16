import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import yaml
import os
import threading
import glob
from PIL import Image, ImageTk
from pipeline import ISPPipeline
import sys
from io import StringIO

class ISPGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISP Pipeline - 图像处理工具")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # 配置数据
        self.config = {}
        self.load_config()
        
        # 界面变量
        self.module_vars = {}
        self.param_vars = {}
        self.sensor_vars = {}
        self.current_image_index = 0
        self.processed_images = []
        
        # 创建界面
        self.create_widgets()
        
        # 重定向stdout到日志
        self.setup_log_redirect()
        
    def setup_log_redirect(self):
        """设置日志重定向"""
        class LogRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, message):
                if message.strip():  # 只记录非空消息
                    self.text_widget.insert(tk.END, message)
                    self.text_widget.see(tk.END)
                    self.text_widget.update()
                    
            def flush(self):
                pass
        
        # 保存原始stdout
        self.original_stdout = sys.stdout
        # 重定向到日志窗口
        sys.stdout = LogRedirector(self.log_text)
        
    def create_widgets(self):
        # 创建主容器
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=1)
        
        # 右侧面板（图像预览和日志）
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=2)
        
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
        
    def create_left_panel(self, parent):
        # 创建滚动区域
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 绑定鼠标滚轮
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 在滚动框架中创建内容
        self.create_file_selection(scrollable_frame)
        self.create_sensor_config(scrollable_frame)
        self.create_isp_modules(scrollable_frame)
        
    def create_file_selection(self, parent):
        """文件选择区域"""
        file_frame = ttk.LabelFrame(parent, text="📁 文件选择", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="选择RAW图像文件夹", 
                  command=self.select_input_folder).pack(pady=5)
        
        self.input_path_var = tk.StringVar()
        path_label = ttk.Label(file_frame, textvariable=self.input_path_var, 
                              wraplength=350, foreground="blue")
        path_label.pack(pady=2)
        
        # 处理控制按钮移到这里
        control_frame = ttk.Frame(file_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="🚀 开始处理", 
                  command=self.start_processing).pack(pady=5)
        
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
    def create_sensor_config(self, parent):
        """传感器配置区域"""
        sensor_frame = ttk.LabelFrame(parent, text="📷 传感器配置", padding=10)
        sensor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 图像尺寸
        size_frame = ttk.Frame(sensor_frame)
        size_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(size_frame, text="宽度:").pack(side=tk.LEFT)
        self.sensor_vars['width'] = tk.StringVar()
        self.sensor_vars['width'].set(str(self.config.get('raw', {}).get('width', 2048)))
        ttk.Entry(size_frame, textvariable=self.sensor_vars['width'], width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(size_frame, text="高度:").pack(side=tk.LEFT, padx=(20,0))
        self.sensor_vars['height'] = tk.StringVar()
        self.sensor_vars['height'].set(str(self.config.get('raw', {}).get('height', 2048)))
        ttk.Entry(size_frame, textvariable=self.sensor_vars['height'], width=8).pack(side=tk.LEFT, padx=5)
        
        # 位深度
        depth_frame = ttk.Frame(sensor_frame)
        depth_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(depth_frame, text="传感器位深:").pack(side=tk.LEFT)
        self.sensor_vars['sensor_bit_depth'] = tk.StringVar()
        self.sensor_vars['sensor_bit_depth'].set(str(self.config.get('raw', {}).get('sensor_bit_depth', 10)))
        depth_combo = ttk.Combobox(depth_frame, textvariable=self.sensor_vars['sensor_bit_depth'], 
                                  values=['8', '10', '12', '14', '16'], width=6, state="readonly")
        depth_combo.pack(side=tk.LEFT, padx=5)
        
    def create_isp_modules(self, parent):
        """ISP模块配置区域"""
        modules_frame = ttk.LabelFrame(parent, text="🔧 ISP模块配置", padding=10)
        modules_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 模块列表（按正确顺序）
        modules = [
            ("DPC (坏点校正)", "dpc", {
                "threshold": 5.0,
                "method": "median"
            }),
            ("鱼眼掩膜", "fisheye_mask", {
                "radius": 1456,
                "center": "[1456, 1456]"
            }),
            ("BLC (黑电平校正)", "blc", {
                "black_level": 64.0
            }),
            ("曝光补偿", "exposure_compensation", {
                "mode": "auto",
                "target_brightness": 0.6,
                "manual_gain": 1.0
            }),
            ("LSC (镜头阴影校正)", "lsc", {
                "strength": 0.3,
                "model_type": "cosine_fourth",
                "focal_length_mm": 4.0,
                "pixel_size_um": 1.4
            }),
            ("White Balance (白平衡)", "wb", {
                "method": "gray_world",
                "r_gain": 1.0,
                "g_gain": 1.0,
                "b_gain": 1.0
            }),
            ("Demosaic (去马赛克)", "demosaic", {
                "method": "frequency_domain",
                "bayer_pattern": "rggb"
            }),
            ("CCM (色彩校正)", "ccm", {
                "saturation_boost": 1.8
            }),
            ("降噪", "denoise", {
                "strength": 1.0,
                "preserve_edges": True
            }),
            ("色度降噪", "chroma_denoise", {
                "strength": 0.8,
                "chroma_strength": 0.7
            }),
            ("阴影高光", "shadow_highlight", {
                "shadow_lift": 0.1,
                "highlight_roll": 0.8
            }),
            ("色调映射", "tonemapping", {
                "method": "reinhard",
                "lift": 0.1,
                "roll": 0.8,
                "brightness": 1.05,
                "contrast": 1.1
            }),
            ("Gamma校正", "gamma", {
                "gamma": 2.2,
                "curve_type": "s_curve"
            }),
            ("锐化", "sharpen", {
                "strength": 1.5,
                "method": "unsharp_masking"
            }),
            ("超分辨率", "super_resolution", {
                "scale_factor": 2.0,
                "method": "bicubic"
            })
        ]
        
        for name, key, params in modules:
            self.create_module_widget(modules_frame, name, key, params)
            
    def create_module_widget(self, parent, name, key, params):
        """创建单个模块的控件"""
        # 模块框架
        module_frame = ttk.LabelFrame(parent, text=name, padding=5)
        module_frame.pack(fill=tk.X, pady=3)
        
        # 启用复选框
        enable_frame = ttk.Frame(module_frame)
        enable_frame.pack(fill=tk.X)
        
        self.module_vars[key] = tk.BooleanVar()
        self.module_vars[key].set(self.config.get(key, {}).get('enable', True))
        
        enable_check = ttk.Checkbutton(enable_frame, text="启用", 
                                      variable=self.module_vars[key])
        enable_check.pack(side=tk.LEFT)
        
        # 参数区域
        if params:
            params_frame = ttk.Frame(module_frame)
            params_frame.pack(fill=tk.X, pady=2)
            
            self.param_vars[key] = {}
            
            for param, default_value in params.items():
                param_frame = ttk.Frame(params_frame)
                param_frame.pack(fill=tk.X, pady=1)
                
                ttk.Label(param_frame, text=f"{param}:", width=15).pack(side=tk.LEFT)
                
                # 获取当前配置值
                current_value = self.config.get(key, {}).get(param, default_value)
                
                # 根据参数类型创建控件
                if param in ['method', 'bayer_pattern', 'curve_type', 'model_type', 'mode']:
                    # 下拉框
                    self.param_vars[key][param] = tk.StringVar()
                    self.param_vars[key][param].set(str(current_value))
                    
                    values = self.get_combo_values(key, param)
                    combo = ttk.Combobox(param_frame, textvariable=self.param_vars[key][param],
                                       values=values, width=15, state="readonly")
                    combo.pack(side=tk.RIGHT)
                    
                elif isinstance(default_value, bool):
                    # 布尔值复选框
                    self.param_vars[key][param] = tk.BooleanVar()
                    self.param_vars[key][param].set(current_value)
                    ttk.Checkbutton(param_frame, variable=self.param_vars[key][param]).pack(side=tk.RIGHT)
                    
                elif isinstance(default_value, (int, float)):
                    # 数值输入框和滑块
                    self.param_vars[key][param] = tk.StringVar()
                    self.param_vars[key][param].set(str(current_value))
                    
                    entry = ttk.Entry(param_frame, textvariable=self.param_vars[key][param], width=8)
                    entry.pack(side=tk.RIGHT)
                    
                    # 为0-1范围的参数添加滑块
                    if isinstance(default_value, float) and 0 <= default_value <= 1:
                        scale = ttk.Scale(param_frame, from_=0, to=1, 
                                        variable=self.param_vars[key][param], 
                                        orient=tk.HORIZONTAL, length=100)
                        scale.pack(side=tk.RIGHT, padx=5)
                        
                else:
                    # 字符串输入框
                    self.param_vars[key][param] = tk.StringVar()
                    self.param_vars[key][param].set(str(current_value))
                    ttk.Entry(param_frame, textvariable=self.param_vars[key][param], width=15).pack(side=tk.RIGHT)
                    
    def create_right_panel(self, parent):
        """右侧面板：图像预览和日志"""
        # 创建垂直分割面板
        right_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # 图像预览区域
        image_frame = ttk.LabelFrame(right_paned, text="🖼️ 图像预览", padding=10)
        right_paned.add(image_frame, weight=2)
        
        # 图像显示
        self.image_label = ttk.Label(image_frame, text="选择图像文件夹后开始处理\n处理完成后将显示结果图像", 
                                    font=("Arial", 12), foreground="gray")
        self.image_label.pack(expand=True)
        
        # 图像导航
        nav_frame = ttk.Frame(image_frame)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="◀ 上一张", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="下一张 ▶", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="🔄 刷新图像", command=self.refresh_images).pack(side=tk.LEFT, padx=5)
        
        self.image_info = ttk.Label(nav_frame, text="", font=("Arial", 10))
        self.image_info.pack(side=tk.RIGHT)
        
        # 日志区域
        log_frame = ttk.LabelFrame(right_paned, text="📋 处理日志", padding=10)
        right_paned.add(log_frame, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 日志控制
        log_control = ttk.Frame(log_frame)
        log_control.pack(fill=tk.X, pady=(5,0))
        
        ttk.Button(log_control, text="清空日志", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(log_control, text="保存日志", command=self.save_log).pack(side=tk.LEFT, padx=5)

    def load_config(self):
        """加载配置文件"""
        if getattr(sys, 'frozen', False):
        # 打包后的exe环境
            config_path = os.path.join(sys._MEIPASS, 'config.yaml')
        else:
        # 开发环境
            config_path = "config.yaml"
            
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                print(f"配置文件已加载: {config_path}")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                self.config = self.get_default_config()
        else:
            print(f"配置文件不存在: {config_path}，使用默认配置")
            self.config = self.get_default_config()
    
    def get_default_config(self):
        """获取默认配置"""
        return {
            'raw': {
                'width': 2048,
                'height': 2048,
                'sensor_bit_depth': 10
            },
            'dpc': {'enable': True, 'threshold': 5.0, 'method': 'median'},
            'fisheye_mask': {'enable': False, 'radius': 1456, 'center': '[1456, 1456]'},
            'blc': {'enable': True, 'black_level': 64.0},
            'exposure_compensation': {'enable': False, 'mode': 'auto', 'target_brightness': 0.6, 'manual_gain': 1.0},
            'lsc': {'enable': True, 'strength': 0.3, 'model_type': 'cosine_fourth', 'focal_length_mm': 4.0, 'pixel_size_um': 1.4},
            'wb': {'enable': True, 'method': 'gray_world', 'r_gain': 1.0, 'g_gain': 1.0, 'b_gain': 1.0},
            'demosaic': {'enable': True, 'method': 'frequency_domain', 'bayer_pattern': 'rggb'},
            'ccm': {'enable': True, 'saturation_boost': 1.8,'highlight_threshold': 0.7,'matrix': [[1.2, -0.1, -0.1],[-0.05, 1.2, -0.15],[0.0, -0.1, 1.1]]},
            'denoise': {'enable': False, 'strength': 1.0, 'preserve_edges': True},
            'chroma_denoise': {'enable': False, 'strength': 0.8, 'chroma_strength': 0.7},
            'shadow_highlight': {'enable': False, 'shadow_lift': 0.1, 'highlight_roll': 0.8},
            'tonemapping': {'enable': True, 'method': 'reinhard', 'lift': 0.1, 'roll': 0.8, 'brightness': 1.05, 'contrast': 1.1},
            'gamma': {'enable': True, 'gamma': 2.2, 'curve_type': 's_curve'},
            'sharpen': {'enable': False, 'strength': 1.5, 'method': 'unsharp_masking'},
            'super_resolution': {'enable': False, 'scale_factor': 2.0, 'method': 'bicubic'}
        }

    def get_combo_values(self, module_key, param):
        """获取下拉框选项"""
        combo_options = {
            'demosaic': {
                'method': ['opencv_vng', 'opencv_ea', 'frequency_domain', 'anti_moire', 'selective_anti_moire'],
                'bayer_pattern': ['rggb', 'bggr', 'grbg', 'gbrg']
            },
            'wb': {
                'method': ['manual', 'gray_world', 'white_patch']
            },
            'gamma': {
                'curve_type': ['standard', 's_curve', 'linear']
            },
            'lsc': {
                'model_type': ['cosine_fourth', 'polynomial', 'radial']
            },
            'tonemapping': {
                'method': ['reinhard', 'aces', 'linear']
            },
            'sharpen': {
                'method': ['unsharp_masking', 'laplacian']
            },
            'super_resolution': {
                'method': ['bicubic', 'bilinear', 'nearest']
            },
            'dpc': {
                'method': ['median', 'mean', 'threshold']
            },
            'exposure_compensation': {
                'mode': ['auto', 'manual']
            }
        }
        return combo_options.get(module_key, {}).get(param, [])

    def select_input_folder(self):
        """选择输入文件夹"""
        folder = filedialog.askdirectory(title="选择包含RAW图像的文件夹")
        if folder:
            self.input_path_var.set(folder)
            self.log(f"已选择输入文件夹: {folder}")
            # 检查文件夹中的RAW文件
            raw_files = glob.glob(os.path.join(folder, "*.raw"))
            if raw_files:
                self.log(f"找到 {len(raw_files)} 个RAW文件")
            else:
                self.log("警告: 未找到RAW文件，请确认文件夹路径和文件扩展名")

    def start_processing(self):
        """开始处理"""
        if not self.input_path_var.get():
            messagebox.showerror("错误", "请先选择输入文件夹")
            return
        
        self.clear_log()
        self.log("开始处理图像...")
        
        # 在新线程中运行处理
        self.progress.start()
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()

    def process_images(self):
        """处理图像（在后台线程中运行）"""
        try:
            # 更新配置
            self.update_config_from_gui()
            
            # 保存临时配置文件
            temp_config_path = "temp_config.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            # 创建ISP流水线
            pipeline = ISPPipeline(temp_config_path)
            
            # 运行处理
            pipeline.run()
            
            # 清理临时文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            self.root.after(0, self.processing_complete)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self.processing_error(error_msg))

    def update_config_from_gui(self):
        """从GUI更新配置"""
        # 更新传感器配置
        self.config['raw'] = {
            'width': int(self.sensor_vars['width'].get()),
            'height': int(self.sensor_vars['height'].get()),
            'sensor_bit_depth': int(self.sensor_vars['sensor_bit_depth'].get()),
            'input_dir': self.input_path_var.get(),
            'raw_to_16bit_scale_factor': 64
        }
        
        # 设置输出目录
        if 'output' not in self.config:
            self.config['output'] = {}
        output_base = self.input_path_var.get()
        self.config['output']['output_dir'] = os.path.join(output_base, 'output')
        self.config['output']['debug_dir'] = os.path.join(output_base, 'debug')
        
        # 更新模块配置
        for module_key in self.module_vars:
            if module_key not in self.config:
                self.config[module_key] = {}
            
            self.config[module_key]['enable'] = self.module_vars[module_key].get()
            
            if module_key in self.param_vars:
                for param_key, var in self.param_vars[module_key].items():
                    value = var.get()
                    # 尝试转换为适当的类型
                    try:
                        if isinstance(var, tk.BooleanVar):
                            value = var.get()
                        elif '.' in str(value) and str(value).replace('.', '').replace('-', '').isdigit():
                            value = float(value)
                        elif str(value).replace('-', '').isdigit():
                            value = int(value)
                        elif str(value).lower() in ['true', 'false']:
                            value = str(value).lower() == 'true'
                    except:
                        pass  # 保持字符串
                    
                    self.config[module_key][param_key] = value

    def processing_complete(self):
        """处理完成回调"""
        self.progress.stop()
        self.log("✅ 处理完成！")
        messagebox.showinfo("完成", "图像处理完成！")
        
        # 自动刷新图像列表
        self.refresh_images()

    def processing_error(self, error_msg):
        """处理错误回调"""
        self.progress.stop()
        self.log(f"❌ 处理错误: {error_msg}")
        messagebox.showerror("错误", f"处理失败: {error_msg}")

    def refresh_images(self):
        """刷新处理后的图像列表"""
        if not self.input_path_var.get():
            return
            
        output_dir = os.path.join(self.input_path_var.get(), 'output')
        if os.path.exists(output_dir):
            # 查找处理后的图像
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            self.processed_images = []
            for ext in image_extensions:
                self.processed_images.extend(glob.glob(os.path.join(output_dir, ext)))
            
            if self.processed_images:
                self.processed_images.sort()
                self.current_image_index = 0
                self.display_image()
                self.log(f"找到 {len(self.processed_images)} 张处理后的图像")
            else:
                self.log("未找到处理后的图像")

    def prev_image(self):
        """上一张图像"""
        if self.processed_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()

    def next_image(self):
        """下一张图像"""
        if self.processed_images and self.current_image_index < len(self.processed_images) - 1:
            self.current_image_index += 1
            self.display_image()

    def display_image(self):
        """显示当前图像"""
        if not self.processed_images:
            return
        
        try:
            image_path = self.processed_images[self.current_image_index]
            image = Image.open(image_path)
            
            # 调整图像大小以适应显示区域
            display_size = (500, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # 保持引用
            
            # 更新图像信息
            info_text = f"{self.current_image_index + 1}/{len(self.processed_images)} - {os.path.basename(image_path)}"
            self.image_info.configure(text=info_text)
            
        except Exception as e:
            self.log(f"显示图像失败: {e}")

    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)

    def save_log(self):
        """保存日志"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))

    def log(self, message):
        """添加日志"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = ISPGUIApp(root)
    root.mainloop()