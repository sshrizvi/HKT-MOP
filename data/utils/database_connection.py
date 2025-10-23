import sys
import os
import logging


# Setting Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, root_dir)


# Setting Up Logger
from logging_config import setup_logging
logger = logging.getLogger('hkt-mop.data.utils')
setup_logging()


# DatabaseConnection Class
class DatabaseConnection():
    pass