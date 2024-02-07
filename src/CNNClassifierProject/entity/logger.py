import logging
import os,sys
from datetime import datetime


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(message)s", #this is the best practice format 
    level=logging.INFO,
)

logger=logging.getLogger("cnnClassifierLogger")
# if __name__=="__main__":
#     try:
#         a=10/0
#     except Exception as e:
#         logging.info(e)
