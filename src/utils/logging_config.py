import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logger = logging.getLogger('stock_analyzer')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    f'logs/stock_analyzer_{datetime.now().strftime("%Y%m%d")}.log'
)

# Create formatters and add it to handlers
log_format = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s'
)
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)