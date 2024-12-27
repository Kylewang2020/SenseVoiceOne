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
    if logLevel > logging.DEBUG:
        formatter= logging.Formatter(
            fmt = '[%(levelname)-5s|%(asctime)s.%(msecs)03d|%(thread)s|%(lineno)03d@%(funcName)-9s]: %(message)s',
            datefmt='%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(
            fmt = '[%(levelname)-5s|%(asctime)s.%(msecs)03d|%(thread)s|%(filename)s:%(lineno)3d@%(funcName)-9s]: %(message)s',
            datefmt = '%m-%d %H:%M:%S')
    handler = logging.FileHandler(LogFileName, encoding="utf-8")
    handler.setFormatter(formatter)
    curLog.addHandler(handler)
    if not isDaemon:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        curLog.addHandler(console)
    logging.debug(f"logger ok. file: {LogFileName}. logLevel:{logLevel}")
