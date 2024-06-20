import os
import logging
import sys


def get_logger(log_dir, name, log_filename="info.log", level=logging.INFO):
    """
    Create and configure a logger object.

    Args:
        log_dir (str): The directory where the log file will be saved.
        name (str): The name of the logger.
        log_filename (str, optional): The name of the log file. Defaults to "info.log".
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger object.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    print("Log directory: %s", log_dir)
    return logger
