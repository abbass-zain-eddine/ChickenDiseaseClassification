import numpy as np
import torch
import os
from PIL import Image
from pathlib import Path
from CNNClassifierProject.config.configuration import PredictConfig
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from CNNClassifierProject.config.configuration import ConfigurationManager
class Prediction():
    def __init__(self,config:PredictConfig,classes:dict):
        self.model_dir= config.path_to_model
        self.model_weights=config.path_to_weights
        self.params_image_size=config.params_image_size
        self.load_model()
        self.classes=classes
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_model(self):
        self.model=torch.load(self.model_dir)
        self.model.load_state_dict(torch.load(self.model_weights))

    def predict(self,image_path:Path=None,images_folder:Path =None,path_output:Path='./'):
        self.test_transorm=T.Compose(
            [
                T.Resize(self.params_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            )
        if image_path:
            test_image=Image.open(image_path)
            test_image=self.test_transorm(test_image)
            test_image=torch.unsqueeze(test_image,0)
            test_image=test_image.to(self.device)
            results=self.model(test_image).cpu().detach().numpy()
            result=np.argmax(results[0],axis=0)
            label=self.classes[result]
            print(result)
            print(label)
            return [{ "image" : label}]

        elif images_folder:
            

            self.dataset=datasets.ImageFolder(images_folder,self.test_transorm)
            self.test_data_generator= DataLoader(self.dataset,8)
            self._predict_batch(path_output)


    def _predict_batch(self,path_output):
        with torch.no_grad():
                for Images in self.test_data_generator:
                    Images = Images.to(self.device)
                    output = self.model(Images)
                    predicted=torch.argmax(output,dim=1)
                    labels=self.classes[predicted]
                    for i in range(predicted.size(0)):
                        with open(os.path.join(path_output,'model_predictions.txt'), 'a') as f:
                            f.write(f'{labels[i]}\n')


if __name__ == '__main__':
    configMngr = ConfigurationManager()
    config=configMngr.get_predict_config()
    classes={0: 'Coccidiosis', 1: 'Healthy', 2: 'Salmonella', 3: 'New Castle Disease'}
    prediction=Prediction(config=config,classes=classes)
    prediction.predict(image_path="artifacts/data_ingestion/data/Train/cocci.3.jpg",path_output=Path('./'))

            

 
