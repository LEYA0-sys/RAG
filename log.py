import os
import sys
import logging
from datetime import datetime

# --- 控制台 + 文件双输出设置 ---
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 创建日志文件夹
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"log_{timestamp}.txt")
# 重定向输出
sys.stdout = DualLogger(log_file)
sys.stderr = sys.stdout  # 捕获异常信息

# 日志模块设置 
logging.basicConfig(level=logging.INFO)
logging.info("日志系统已初始化")
