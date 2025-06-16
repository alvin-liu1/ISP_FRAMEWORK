# 文件：main.py
# ---------------------
# 主程序入口：加载 config.yaml，执行 ISP 流程
# 作为新手：你可以在这里一键运行整条流水线流程。
from pipeline import ISPPipeline


if __name__ == "__main__":
    print("main 启动中...")

    isp = ISPPipeline("config.yaml")
    isp.run()