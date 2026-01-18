"""Logging utilities."""

import logging
import os
from datetime import datetime
import colorlog


def setup_logger(log_dir, rank=0):
    """
    Setup colored logger.
    
    Args:
        log_dir: Directory to save logs
        rank: Process rank (for distributed training)
    
    Returns:
        logger: Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('UTI_Generation')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    console_format = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s][Rank %(rank)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'train_rank{rank}_{timestamp}.log')
    )
    file_handler.setLevel(logging.INFO)
    
    file_format = logging.Formatter(
        '[%(asctime)s][Rank %(rank)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Add rank as extra
    logger = logging.LoggerAdapter(logger, {'rank': rank})
    
    return logger


if __name__ == "__main__":
    logger = setup_logger('logs', rank=0)
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
