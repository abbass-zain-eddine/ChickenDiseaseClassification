from src.CNNClassifierProject.config.configuration import ConfigurationManager
from src.CNNClassifierProject.components.trainer import Trainer
from src.CNNClassifierProject.data.dataset import CustomDataset
from src.CNNClassifierProject.utils.common import train_val_split
from torchvision import transforms as T
import torch

STAGE_NAME="Training"

class ModelTrainingPipline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()

        training_config=config.get_training_configs()

        base_model_config=config.get_base_model_config
        train_images,test_images,train_labels,test_labels=train_val_split(training_config.training_csv_dir,training_config.training_data_dir)
        train_transorm=T.Compose(
            [
                T.Resize(training_config.params_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.2),
                T.RandomRotation(10,interpolation=T.InterpolationMode.BILINEAR),
                T.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
                T.OneOf([
                    T.Blur(blur_limit=3, p=0.2),
                    T.MedianBlur(blur_limit=3, p=0.2),
                ], p=0.4),
                T.OneOf([
                    T.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=0.5),
                    T.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5)
                ], p=0.4),

            ]
        )
        val_transorm=T.Compose(
            [
                T.Resize(training_config.params_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        train_dataset= CustomDataset(train_images,train_labels)
        val_dataset= CustomDataset(test_images,test_labels)
        

        training=Trainer(base_model_config, training_config,)