#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISP Pipeline GUI启动器
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """检查依赖包"""
    missing = []
    
    try:
        import tkinter
    except ImportError:
        missing.append("tkinter (Python内置，可能需要重新安装Python)")
    
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing.append("pillow")
    
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    return missing

if __name__ == "__main__":
    print("检查依赖包...")
    missing = check_dependencies()
    
    if missing:
        print("缺少以下依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n请运行以下命令安装:")
        print("pip install pillow pyyaml opencv-python numpy scipy rawpy")
        input("按回车键退出...")
        sys.exit(1)
    
    try:
        from gui_app import ISPGUIApp
        import tkinter as tk
        
        print("启动ISP Pipeline GUI...")
        root = tk.Tk()
        app = ISPGUIApp(root)
        root.mainloop()
        
    except Exception as e:
        print(f"启动错误: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
