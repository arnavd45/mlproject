import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs")  # CHANGED: Removed LOG_FILE from here
os.makedirs(logs_path,exist_ok=True)  # Creates just the "logs" folder

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)  # Creates path: logs/filename.log

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
