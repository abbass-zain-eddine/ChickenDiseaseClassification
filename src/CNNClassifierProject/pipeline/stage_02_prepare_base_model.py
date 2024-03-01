from CNNClassifierProject.config.configuration import ConfigurationManager
from CNNClassifierProject.components.prepare_base_model import PrepareBaseModel
from CNNClassifierProject.entity.logger import logging
from CNNClassifierProject.config.model_head_configurations import custom_layers_params,custom_layers_types

STAGE_NAME= "prepare_base_model"

class PrepareBaseModelTrainingPipline:
    def __init__(self):
        pass

    def main(self):
        config= ConfigurationManager()
        prepare_base_model_config= config.get_prepare_base_model_config()
        prepare_base_model= PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.prepare_full_model(custom_layers_types, custom_layers_params)
        prepare_base_model.save_model()



if __name__ == "__main__":
    try:
        logging.info("***************************")
        logging.info(STAGE_NAME)
        obj=PrepareBaseModelTrainingPipline()
        obj.main()
        logging.info(f">>>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e