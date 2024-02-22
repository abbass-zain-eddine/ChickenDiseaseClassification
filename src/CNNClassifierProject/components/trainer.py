from CNNClassifierProject.entity.config_entity import PrepareBaseModelConfig, TrainingConfigs
from CNNClassifierProject.config.model_head_configurations import (custom_layers_params, 
                                                                   custom_layers_types,
                                                                   loss_fn,
                                                                   metrix,
                                                                   optimizer,
                                                                   learning_rate,
                                                                     nb_classes,
                                                                     check_points_dir)
from src.CNNClassifierProject.utils.common import create_dir
import torch
from pathlib import Path
from torch import nn
from CNNClassifierProject.components.prepare_base_model import PrepareBaseModel
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
class Trainer():
    def __init__(self, base_model_config: PrepareBaseModelConfig, training_config: TrainingConfigs, train_Loader, val_loader):
        self.base_model_config = base_model_config()
        self.custom_layers_types = custom_layers_types
        self.training_config=training_config
        self.custom_layers_params = custom_layers_params
        self.nb_classes = nb_classes
        self.classes = train_Loader.dataset.get_idx_to_class()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pbm=PrepareBaseModel(self.base_model_config)
        self.pbm.get_base_model()
        self.pbm.prepare_full_model(self.custom_layers_types, self.custom_layers_params)
        self.pbm.save_model()
        self.model = self.pbm.updated_model.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(),lr=self.learning_rate)
        self.loss_fn = loss_fn
        self.metrix = metrix
        self.num_epochs = self.training_config.params_epochs
        self.train_Loader = train_Loader
        self.val_loader = val_loader
        self.TBWriter=SummaryWriter()
        


    def train_step(self,epoch,log_interval=10):
        self.model.train()
        running_loss=0.0
        correct=0
        total=0
        running_train_loss = []
        for batch_idx, (Images, targets) in enumerate(self.train_Loader):
            batch_start_time = time.time()
            Images=Images.to(self.device)
            targets=targets.to(self.device)
            output=self.model(Images)
            self.optimizer.zero_grad()

            loss=loss_fn(output,targets)
            running_train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            running_loss = running_loss + loss.item()
            
            predicted=torch.argmax(output,dim=1)
            #labels=targets#torch.argmax(targets,dim=1)
            total+=targets.size(0)
            correct+=(targets == predicted).sum().item()
            
            if (batch_idx + 1)%10 == 0:
                batch_time = time.time() - batch_start_time
                m,s = divmod(batch_time, 60)
                print('train loss @batch_idx {}/{}: {} in {} mins {} secs (per batch)'.format(str(batch_idx+1).zfill(len(str(len(self.train_Loader)))), len(self.train_Loader), loss.item(), int(m), round(s, 2)))

            if((batch_idx-1) % log_interval == 0):             
                self.TBWriter.add_images("images", Images[0], global_step=epoch*len(self.train_Loader)+batch_idx, dataformats='CHW')
                true_label_name=str(self.classes[targets.detach().clone().cpu()[0].item()])
                predicted_label_name=str(self.classes[predicted.detach().clone().cpu()[0].item()])
                label_text = f'True label: {true_label_name}\nPredicted label: {predicted_label_name}'
                self.TBWriter.add_text(f'labels_{batch_idx}', label_text)

        return np.array(running_train_loss).mean(),correct,total,running_loss
    




    def val_step(self,epoch,log_interval=10):
        running_val_loss=[]
        self.model.eval()
        with torch.no_grad():
            totalval=0
            correctval=0
            for batch_idx, (Images,targets) in enumerate(self.val_loader):
                Images = Images.to(self.device)
                targets = targets.to(self.device)
                output = self.model(Images)
                #targets=torch.vstack(targets)
                #targets=targets.transpose(0,1).float().to(self.device)
                loss = loss_fn(output, targets)
                running_val_loss.append(loss.item())
                #_,predicted=torch.max(output.data,1)
                #predicted=torch.argmax(predicted,dim=1)
                predicted=torch.argmax(output,dim=1)
                #labels=targets#torch.argmax(targets,dim=1)
                totalval+=targets.size(0)
                correctval+=(predicted == targets).sum().item()               
                if((batch_idx-1) % 10 == 0):
                    self.TBWriter.add_images("images", Images[0], global_step=epoch*len(self.train_Loader)+batch_idx, dataformats='CHW')
                    #TB.add_text("class", str(im_class(targets,["Speckle_Noise","Salt_Pepper","Uneven_Illumination"])), global_step=epoch*len(train_Loader)+batch_idx)
                    true_label_name=str(self.classes[targets.detach().clone()[0].item()])
                    predicted_label_name=str(self.classes[predicted.detach().clone()[0].item()])
                    label_text = f'True label: {true_label_name[0]}\nPredicted label: {predicted_label_name[0]}'
                    self.TBWriter.add_text(f'labels_{batch_idx}', label_text)
                if (batch_idx + 1)%log_interval == 0:
                    print('val loss   @batch_idx {}/{}: {}'.format(str(batch_idx+1).zfill(len(str(len(self.val_loader)))), len(self.val_loader), loss.item()))
        return np.array(running_val_loss).mean(),correctval,totalval,
    
    def train(self,log_dir="/logs_dir",log_interval=10):

        self.log_dir=os.path.join(self.training_config.root_dir,log_dir)
        self.check_points_dir=os.path.join(self.training_config.root_dir,check_points_dir)
        #create_dir([Path(self.log_dir)])
        create_dir([Path(self.check_points_dir)])
        
        script_time = time.time()
        train_epoch_loss = []
        val_epoch_loss = []

        
        print('\nTRAINING...')
        epoch_train_start_time = time.time()
        
        best_acc=0
        for epoch in range(self.num_epochs):
            
            running_train_loss_mean,correct,total,running_loss=self.train_step(epoch,log_interval=log_interval)

            accuracy=100*correct/total
            train_epoch_loss.append(running_train_loss_mean)
            self.TBWriter.add_scalar("Loss/epoch", running_train_loss_mean, epoch)
            epoch_train_time = time.time() - epoch_train_start_time
            m,s = divmod(epoch_train_time, 60)
            h,m = divmod(m, 60)
            print('\nepoch train time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))
            print('\nepoch {} train acc: {}'.format(int(epoch),float(accuracy)))

            
            print('\nVALIDATION...')
            epoch_val_start_time = time.time()

            running_val_loss_mean,correctval,totalval=self.val_step(epoch,log_interval=log_interval)

            val_epoch_loss.append(running_val_loss_mean)
            accval=100*correctval/totalval
            epoch_val_time = time.time() - epoch_val_start_time
            m,s = divmod(epoch_val_time, 60)
            h,m = divmod(m, 60)
            print('\nepoch val   time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

            train_loss=running_loss/len(self.train_Loader)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
            print('\nepoch val acc: {}'.format(float(accval)))

            if accval>=best_acc:
                torch.save(self.model.state_dict(),'artifacts/training/check_points/best_model.pth')
                best_acc=accval
            torch.save(self.model.state_dict(),os.path.join(self.check_points_dir,'last_model.pth'))
        total_script_time = time.time() - script_time
        m, s = divmod(total_script_time, 60)
        h, m = divmod(m, 60)
        print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
        
        print('\nFin.')






        