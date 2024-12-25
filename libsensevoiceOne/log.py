import logging
import os

# 日志初始化
def logInit(LogFileName="one.log", isDaemon=False, logLevel=logging.DEBUG):
    logPath = "./log"
    if not os.path.exists(logPath): 
        os.makedirs(logPath)
    LogFileName = os.path.join(logPath, LogFileName)
    curLog = logging.getLogger()
    curLog.setLevel(level = logLevel)
    formatter = logging.Formatter(
        fmt = '[%(levelname)-5s|%(asctime)s.%(msecs)03d|%(filename)s-%(funcName)s:%(lineno)3d] %(message)s',
        datefmt = '%m-%d %H:%M:%S')
    handler = logging.FileHandler(LogFileName, encoding="utf-8")
    handler.setFormatter(formatter)
    curLog.addHandler(handler)
    if not isDaemon:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        curLog.addHandler(console)
    logging.debug("logger is ready")
