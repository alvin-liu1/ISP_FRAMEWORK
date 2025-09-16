import PyInstaller.__main__
import os

# 打包配置
PyInstaller.__main__.run([
    'run_gui.py',
    '--onefile',
    '--windowed',
    '--name=ISP_Pipeline_GUI',
    '--icon=icon.ico',  # 如果有图标文件
    '--add-data=config.yaml;.',
    '--hidden-import=PIL._tkinter_finder',
    '--hidden-import=cv2',
    '--hidden-import=rawpy',
    '--collect-all=scipy',
    '--collect-all=numpy',
])