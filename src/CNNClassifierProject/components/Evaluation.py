from CNNClassifierProject.entity.config_entity import EvaluationConfig
from CNNClassifierProject.utils.common import write_json
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from pathlib import Path
import torch
from CNNClassifierProject.config.model_head_configurations import loss_fn

class Evaluation:

    def __init__(self,config: EvaluationConfig):
        self.config = config
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = loss_fn

    def _valid_generator(self):
        self.val_transorm=T.Compose(
            [
                T.Resize(self.config.params_image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.dataflow_kwargs=dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        self.dataset=datasets.ImageFolder(self.config.training_data,self.val_transorm)
        

        self.valid_data_generator= DataLoader(self.dataset,8)

    def load_model(self,path:Path):
        self.model=torch.load(self.config.path_of_base_update_model)
        self.model.load_state_dict(torch.load(path))

    


    def evaluate(self):
        running_val_loss=[]
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            totalval=0
            correctval=0
            total_loss=0
            for (Images,targets) in self.valid_data_generator:
                Images = Images.to(self.device)
                targets = targets.to(self.device)
                output = self.model(Images)
                #output = torch.argmax(output,dim=1)
                loss = self.loss_fn(output, targets)
                running_val_loss.append(loss.item())
                predicted=torch.argmax(output,dim=1)
                totalval+=targets.size(0)
                total_loss+=loss.item()
                correctval+=(predicted == targets).sum().item() 

             
            accuracy=correctval/totalval
            avg_loss= total_loss/totalval

            return avg_loss,accuracy
    

    def evaluation(self):
        self.load_model(self.config.path_of_trained_model)
        self._valid_generator()
        self.score=self.evaluate()

    def save_score(self):
        score ={ "loss":self.score[0], "accuracy": self.score[1]}
        write_json(Path("score.json"),score)

