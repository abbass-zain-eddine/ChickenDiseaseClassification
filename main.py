from src.CNNClassifierProject.entity.logger import logging
from src.CNNClassifierProject.pipline.stage_01_data_ingestion import DataIngestionTrainingPipline


STAGE_NAME="stage_01_data_ingestion"
if __name__ == '__main__':
    try:
        logging.info(f" Stage Name: {STAGE_NAME} start")
        obj= DataIngestionTrainingPipline()
        obj.main()
        logging.info(f" Stage Name: {STAGE_NAME} end")
    except Exception as e:
        logging.info(e)
        raise e
