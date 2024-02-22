from  CNNClassifierProject.constants import *
from CNNClassifierProject.utils.common import read_yaml, create_dir
from CNNClassifierProject.entity.config_entity import DataIngestionConfig,   PrepareBaseModelConfig, TrainingConfigs
import os
from pathlib import Path

class ConfigurationManager:

    def __init__(
        self,
        config_filepath= CONFIG_FILE_PATH,
        params_filepath= PARAMS_FILE_PATH):

        self.config= read_yaml(config_filepath)
        self.params= read_yaml(params_filepath)

        create_dir([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        data_ingestion_config= DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_prepare_base_model_config(self)-> PrepareBaseModelConfig:
        config=self.config.prepare_base_model
        create_dir([config.root_dir])

        prepare_base_model_config= PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_dir=Path(config.base_model_dir),
            updated_base_model_dir=Path(config.update_base_model_dir),
            
            params_image_size=self.params.IMAGE_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            param_freeze_layers=self.params.FREEZE_LAYERS
        )
        return prepare_base_model_config
    
    def get_training_configs(self)-> TrainingConfigs:
        training=self.config.training
        prepare_base_model= self.config.prepare_base_model
        params= self.params
        training_data_dir= os.path.join(self.config.data_ingestion.unzip_dir,"Train")
        training_data_csv_dir= os.path.join(self.config.data_ingestion.unzip_dir,"train_data.csv")
        create_dir([Path(training.root_dir)])

        training_configs= TrainingConfigs(
            root_dir=Path(training.root_dir),
            updated_base_model_dir=Path(prepare_base_model.update_base_model_dir),
            training_data_dir=Path(training_data_dir),
            training_csv_dir=Path(training_data_csv_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE
        )
        return training_configs
