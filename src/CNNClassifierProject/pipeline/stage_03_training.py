from CNNClassifierProject.config.configuration import ConfigurationManager
from CNNClassifierProject.components.trainer import Trainer
from CNNClassifierProject.data.dataset import CustomDataset
from CNNClassifierProject.utils.common import train_val_split
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from CNNClassifierProject.entity.logger import logging
STAGE_NAME="Training"

class ModelTrainingPipline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()

        training_config=config.get_training_configs()

        base_model_config=config.get_prepare_base_model_config
        train_images,val_images,train_labels,val_labels=train_val_split(training_config.training_csv_dir,training_config.training_data_dir)
        train_transorm=T.Compose(
            [
                T.Resize(training_config.params_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomHorizontalFlip(0.2),
                T.RandomVerticalFlip(0.2),
               # T.RandomRotation(10,interpolation=T.InterpolationMode.BILINEAR),
            ]
        )
        val_transorm=T.Compose(
            [
                T.Resize(training_config.params_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        train_dataset= CustomDataset(train_images,train_labels,train_transorm)
        val_dataset= CustomDataset(val_images,val_labels,val_transorm)
        train_dataloader = DataLoader(train_dataset, batch_size=training_config.params_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=training_config.params_batch_size, shuffle=True)
        training=Trainer(base_model_config, training_config,train_dataloader,val_dataloader)
        training.train()
        


if __name__ == "__main__":
    try:
        logging.info("***************************")
        logging.info(STAGE_NAME)
        model_trainer=ModelTrainingPipline()
        model_trainer.main()
        logging.info(f">>>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e