import os 
import urllib.request as request
from zipfile import ZipFile
import torchvision.models as models
from CNNClassifierProject.entity.config_entity import PrepareBaseModelConfig
import torch
from torch import nn
import torchvision.transforms as transforms
from CNNClassifierProject.components.Model_creator import Model
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    def get_base_model(self):
        self.model=models.vgg16(weights=self.config.params_weights)
        if self.config.params_include_top:
            self.model=torch.nn.Sequential(*list(self.model.features.children()))
    
    def input_processing(self):
        self.transform=transforms.Compose([
            transforms.Resize(self.config.params_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_full_model(self,custom_layers_types,custom_layers_params):
        freeze_layers= self.config.param_freeze_layers
        if freeze_layers:
            for layer in self.model.features.children():
                layer.requires_grad=False        
        self.updated_model= Model(self.model,custom_layers_types,custom_layers_params)

    def save_model(self):
        torch.save(self.model, self.config.base_model_dir)
        torch.save(self.updated_model, self.config.updated_base_model_dir)


    