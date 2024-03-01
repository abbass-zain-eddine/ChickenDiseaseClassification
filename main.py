from src.CNNClassifierProject.entity.logger import logging
from src.CNNClassifierProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipline
from src.CNNClassifierProject.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipline
from src.CNNClassifierProject.pipeline.stage_03_training import ModelTrainingPipline
from src.CNNClassifierProject.pipeline.stage_04_evaluation import ModelEvaluationPipline
STAGE_NAME_01="stage_01_data_ingestion"
STAGE_NAME_02= "prepare_base_model"
STAGE_NAME_03="stage_03_model_training"
STAGE_NAME_04="stage_04_model_evaluation"

if __name__ == "__main__":
    
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
    
    # try:
    #     logging.info("***************************")
    #     logging.info(STAGE_NAME_03)
    #     model_trainer=ModelTrainingPipline()
    #     model_trainer.main()
    #     logging.info(f">>>>>>>>>>> Stage {STAGE_NAME_03} Completed <<<<<<<<<<<")
    # except Exception as e:
    #     logging.exception(e)
    #     raise e
    
    try:
        logging.info("***************************")
        logging.info(STAGE_NAME_04)
        model_evaluation=ModelEvaluationPipline()
        model_evaluation.main()
        logging.info(f">>>>>>>>>>> Stage {STAGE_NAME_04} Completed <<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e