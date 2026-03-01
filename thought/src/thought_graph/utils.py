import logging
import colorlog

def get_logger(name: str, console_level=logging.INFO, file_level=logging.ERROR) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name (str): The name of the logger.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create console handler with debug level
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(console_level)
    
    # Create file handler with error level
    file_handler = logging.FileHandler(f"log/{name}.log", encoding="utf-8")
    file_handler.setLevel(file_level)
    
    # Create formatter and set it for both handlers
    formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)s] %(message)s",
    log_colors={
        'DEBUG':    'green',    # 蓝色
        'INFO':     'blue',   # 顺便给其他级别也上个色
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger