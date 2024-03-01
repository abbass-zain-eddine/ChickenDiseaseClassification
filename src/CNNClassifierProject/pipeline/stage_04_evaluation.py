from CNNClassifierProject.config.configuration import ConfigurationManager
from CNNClassifierProject.components.Evaluation import Evaluation
from CNNClassifierProject.data.dataset import CustomDataset
from CNNClassifierProject.utils.common import train_val_split
from torchvision import transforms as T
from CNNClassifierProject.entity.logger import logging
import torch

STAGE_NAME="Evaluation"

class ModelEvaluationPipline:
    def __init__(self):
        pass

    def main(self):
        try:
            config=ConfigurationManager()

            val_config=config.get_evaluation_configs()

            evaluation=Evaluation(val_config)
            evaluation.evaluation()
            evaluation.save_score()
        except Exception as e:
            logging.exception(e)
            raise e
        

if __name__ == "__main__":
    try:
        logging.info("***************************")
        logging.info(STAGE_NAME)
        model_evaluation=ModelEvaluationPipline()
        model_evaluation.main()
        logging.info(f">>>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e