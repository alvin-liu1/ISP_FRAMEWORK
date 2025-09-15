import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import yaml
from PIL import Image, ImageTk
import numpy as np
from pipeline import ISPPipeline
import cv2

class ISPGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISP Pipeline - 图像处理工具")
        self.root.geometry("1200x800")
        
        # 配置数据
        self.config = {}
        self.load_config()
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 右侧图像显示
        image_frame = ttk.LabelFrame(main_frame, text="图像预览")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(control_frame)
        self.create_image_panel(image_frame)
        
    def create_control_panel(self, parent):
        # 文件选择
        file_frame = ttk.LabelFrame(parent, text="文件选择")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="选择RAW图像文件夹", 
                  command=self.select_input_folder).pack(pady=5)
        
        self.input_path_var = tk.StringVar()
        ttk.Label(file_frame, textvariable=self.input_path_var, 
                 wraplength=350).pack(pady=2)
        
        # ISP模块控制
        modules_frame = ttk.LabelFrame(parent, text="ISP模块")
        modules_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建滚动框架
        canvas = tk.Canvas(modules_frame, height=300)
        scrollbar = ttk.Scrollbar(modules_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.create_module_controls(scrollable_frame)
        
        # 处理控制
        process_frame = ttk.LabelFrame(parent, text="处理控制")
        process_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(process_frame, text="开始处理", 
                  command=self.start_processing).pack(pady=5)
        
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # 日志输出
        log_frame = ttk.LabelFrame(parent, text="处理日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def create_module_controls(self, parent):
        self.module_vars = {}
        self.param_vars = {}
        
        # 主要ISP模块
        modules = [
            ("DPC (坏点校正)", "dpc", {"threshold": 5.0}),
            ("BLC (黑电平校正)", "blc", {"black_level": 64}),
            ("LSC (镜头阴影校正)", "lsc", {"strength": 0.3}),
            ("Demosaic (去马赛克)", "demosaic", {"method": "selective_anti_moire"}),
            ("White Balance (白平衡)", "wb", {"r_gain": 1.0, "g_gain": 1.0, "b_gain": 1.0}),
            ("CCM (色彩校正)", "ccm", {}),
            ("Gamma校正", "gamma", {"gamma": 2.2}),
            ("色调映射", "tonemapping", {"lift": 0.1, "roll": 0.8}),
            ("锐化", "sharpen", {"strength": 1.5}),
        ]
        
        for name, key, params in modules:
            frame = ttk.LabelFrame(parent, text=name)
            frame.pack(fill=tk.X, padx=2, pady=2)
            
            # 启用/禁用复选框
            var = tk.BooleanVar()
            var.set(self.config.get(key, {}).get('enable', True))
            self.module_vars[key] = var
            
            ttk.Checkbutton(frame, text="启用", variable=var).pack(anchor=tk.W)
            
            # 参数控制
            self.param_vars[key] = {}
            for param, default in params.items():
                param_frame = ttk.Frame(frame)
                param_frame.pack(fill=tk.X, padx=5)
                
                ttk.Label(param_frame, text=f"{param}:").pack(side=tk.LEFT)
                
                param_var = tk.StringVar()
                param_var.set(str(self.config.get(key, {}).get(param, default)))
                self.param_vars[key][param] = param_var
                
                ttk.Entry(param_frame, textvariable=param_var, width=10).pack(side=tk.RIGHT)
    
    def create_image_panel(self, parent):
        # 图像显示区域
        self.image_label = ttk.Label(parent, text="选择图像文件夹后开始处理")
        self.image_label.pack(expand=True)
        
        # 图像切换控制
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT)
        
        self.image_info = ttk.Label(nav_frame, text="")
        self.image_info.pack(side=tk.RIGHT)
        
        self.current_image_index = 0
        self.processed_images = []
    
    def select_input_folder(self):
        folder = filedialog.askdirectory(title="选择RAW图像文件夹")
        if folder:
            self.input_path_var.set(folder)
            self.update_config_paths(folder)
    
    def update_config_paths(self, input_folder):
        self.config['raw']['input_dir'] = input_folder
        self.config['output']['output_dir'] = os.path.join(input_folder, "output")
        self.config['output']['debug_dir'] = os.path.join(input_folder, "debug")
    
    def load_config(self):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except:
            # 默认配置
            self.config = {
                'raw': {'input_dir': '', 'width': 2048, 'height': 2048},
                'dpc': {'enable': True, 'threshold': 5.0},
                'blc': {'enable': True, 'black_level': 64},
                'demosaic': {'enable': True, 'method': 'selective_anti_moire', 'bayer_pattern': 'rggb'},
                'output': {'output_dir': 'output', 'debug_dir': 'debug'}
            }
    
    def save_config(self):
        # 更新配置
        for key, var in self.module_vars.items():
            if key not in self.config:
                self.config[key] = {}
            self.config[key]['enable'] = var.get()
            
            # 更新参数
            if key in self.param_vars:
                for param, param_var in self.param_vars[key].items():
                    try:
                        value = float(param_var.get())
                        self.config[key][param] = value
                    except:
                        self.config[key][param] = param_var.get()
        
        # 保存到文件
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def start_processing(self):
        if not self.input_path_var.get():
            messagebox.showerror("错误", "请先选择输入文件夹")
            return
        
        self.save_config()
        self.progress.start()
        self.log_text.delete(1.0, tk.END)
        
        # 在新线程中处理
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()
    
    def process_images(self):
        try:
            # 重定向输出到GUI
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            # 运行ISP处理
            isp = ISPPipeline("config.yaml")
            isp.run()
            
            # 获取输出
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # 更新GUI
            self.root.after(0, self.processing_complete, output)
            
        except Exception as e:
            self.root.after(0, self.processing_error, str(e))
    
    def processing_complete(self, output):
        self.progress.stop()
        self.log_text.insert(tk.END, output)
        self.log_text.see(tk.END)
        
        # 加载处理结果
        self.load_processed_images()
        messagebox.showinfo("完成", "图像处理完成！")
    
    def processing_error(self, error):
        self.progress.stop()
        self.log_text.insert(tk.END, f"错误: {error}\n")
        messagebox.showerror("处理错误", error)
    
    def load_processed_images(self):
        output_dir = self.config.get('output', {}).get('output_dir', 'output')
        if os.path.exists(output_dir):
            self.processed_images = [f for f in os.listdir(output_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))]
            if self.processed_images:
                self.current_image_index = 0
                self.show_current_image()
    
    def show_current_image(self):
        if not self.processed_images:
            return
        
        output_dir = self.config.get('output', {}).get('output_dir', 'output')
        image_path = os.path.join(output_dir, self.processed_images[self.current_image_index])
        
        try:
            # 加载并缩放图像
            image = Image.open(image_path)
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # 保持引用
            
            # 更新信息
            info = f"{self.current_image_index + 1}/{len(self.processed_images)} - {self.processed_images[self.current_image_index]}"
            self.image_info.configure(text=info)
            
        except Exception as e:
            self.image_label.configure(text=f"无法加载图像: {e}")
    
    def prev_image(self):
        if self.processed_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()
    
    def next_image(self):
        if self.processed_images and self.current_image_index < len(self.processed_images) - 1:
            self.current_image_index += 1
            self.show_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ISPGUIApp(root)
    root.mainloop()