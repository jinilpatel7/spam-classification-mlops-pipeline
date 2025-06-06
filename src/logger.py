import os
import sys
import logging
from datetime import datetime

def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a logger instance that writes logs to a specific directory based on the module name.
    Example: logs/data_ingestion/04_06_2025_16_42_00.log
    """
    # Create log directory for the module
    log_subdir = os.path.join("logs", module_name)
    os.makedirs(log_subdir, exist_ok=True)

    # Log file name with timestamp
    log_file = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
    log_path = os.path.join(log_subdir, log_file)

    # Create a new logger for the module
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if function is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

        # File handler - Create Logging in files
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        # Stream (console) handler - Show realtime logs in terminal
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger

# Optional for test run
if __name__ == "__main__":
    test_logger = get_logger("test_module")
    test_logger.info("This is a test log from test_module")
