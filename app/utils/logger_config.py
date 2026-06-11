"""
日志配置模块
提供统一的日志记录配置和工具函数
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class LoggerConfig:
    """日志配置管理器"""
    
    @staticmethod
    def setup_logger(name: str = "clip_image_search",
                    level: str = "INFO",
                    log_file: Optional[str] = None,
                    console_output: bool = True,
                    file_output: bool = True) -> logging.Logger:
        """设置日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: 日志文件路径，默认为当前日期的日志文件
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            
        Returns:
            配置好的日志记录器
        """
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # 清除已有处理器 + 禁止向 root logger 传播，避免重复
        logger.handlers.clear()
        logger.propagate = False
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件输出处理器
        if file_output:
            if log_file is None:
                # 创建logs目录
                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                
                # 使用当前日期作为日志文件名
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_file = os.path.join(log_dir, f"{name}_{current_date}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
        """记录函数调用信息
        
        Args:
            logger: 日志记录器
            func_name: 函数名称
            **kwargs: 函数参数
        """
        params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.debug(f"调用函数: {func_name}({params})")
    
    @staticmethod
    def log_performance(logger: logging.Logger, operation: str, duration: float, **metrics):
        """记录性能信息
        
        Args:
            logger: 日志记录器
            operation: 操作名称
            duration: 执行时间（秒）
            **metrics: 其他性能指标
        """
        metric_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"性能统计 - {operation}: 耗时={duration:.2f}s, {metric_str}")
    
    @staticmethod
    def log_error_with_context(logger: logging.Logger, error: Exception, context: dict = None):
        """记录带上下文的错误信息
        
        Args:
            logger: 日志记录器
            error: 异常对象
            context: 上下文信息字典
        """
        error_msg = f"错误类型: {type(error).__name__}, 错误信息: {str(error)}"
        
        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            error_msg += f", 上下文: {context_str}"
        
        logger.error(error_msg, exc_info=True)


def get_logger(name: str = "clip_image_search", level: str = "INFO") -> logging.Logger:
    """获取配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        
    Returns:
        日志记录器
    """
    return LoggerConfig.setup_logger(name=name, level=level)


# 创建默认日志记录器
default_logger = get_logger()


def log_execution_time(func):
    """装饰器：记录函数执行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            default_logger.info(f"函数 {func.__name__} 执行完成，耗时: {duration:.2f}秒")
            return result
        except Exception as e:
            duration = time.time() - start_time
            default_logger.error(f"函数 {func.__name__} 执行失败，耗时: {duration:.2f}秒，错误: {e}")
            raise
    
    return wrapper


def log_method_calls(cls):
    """类装饰器：为类的所有公共方法添加日志记录
    
    Args:
        cls: 被装饰的类
        
    Returns:
        装饰后的类
    """
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_execution_time(attr))
    return cls