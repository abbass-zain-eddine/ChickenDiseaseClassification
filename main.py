from src.CNNClassifierProject.entity.logger import logging
from src.CNNClassifierProject.pipline.stage_01_data_ingestion import DataIngestionTrainingPipline
from src.CNNClassifierProject.pipline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipline

STAGE_NAME_01="stage_01_data_ingestion"
STAGE_NAME_02= "prepare_base_model"
if __name__ == '__main__':
    try:
        logging.info(f" Stage Name: {STAGE_NAME_01} start")
        obj= DataIngestionTrainingPipline()
        obj.main()
        logging.info(f" Stage Name: {STAGE_NAME_01} end")
    except Exception as e:
        logging.info(e)
        raise e


    try:
        logging.info("***************************")
        logging.info(STAGE_NAME_02)
        obj=PrepareBaseModelTrainingPipline()
        obj.main()
        logging.info(f">>>>>>>>>>> Stage {STAGE_NAME_02} Completed <<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e