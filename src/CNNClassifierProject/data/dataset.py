import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
import cv2
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self,dataset,labels,transform=None):
        super(CustomDataset, self).__init__()
        self.dataset=dataset
        self.labels=labels
        self.transform=transform
        self.get_idx_to_class()
        self.get_class_to_idx()

    def get_idx_to_class(self):
        self.idx_to_class={key : value for key,value in enumerate(set(self.labels))}
        return self.idx_to_class
    def get_class_to_idx(self):
        self.class_to_idx={value : key for key,value in enumerate(set(self.labels))}
        return self.class_to_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return iter(self.dataset)
    
    def __getitem__(self, idx):
        image_path,label =self.dataset.iloc[idx],self.class_to_idx[self.labels.iloc[idx]]
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image=self.transform(image)
        return image,label

    
