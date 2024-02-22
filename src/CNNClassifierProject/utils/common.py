import os 
import yaml
from box.exceptions import BoxValueError
from CNNClassifierProject.entity.logger import logging
import json
import joblib
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
import base64
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """
    this function reads a yaml file and returns a ConfigBox object
    Args:
    path_to_yaml (Path): path to the yaml file

    Returns:
    ConfigBox: ConfigBox object

    """
    try:
        with open(path_to_yaml, "r") as f:
            config = yaml.safe_load(f)
            logging.info(f"config loaded from {path_to_yaml}")
            return ConfigBox(config)
    except BoxValueError as e:
        logging.info(e)
        raise ValueError("yaml file is empty")
    except FileNotFoundError as e:
        logging.info(e)
        raise FileExistsError("yaml file does not exist")
    except Exception as e:
        logging.info(e)
        raise e
    
@ensure_annotations
def create_dir(path_to_dirs:list,verbose=True):
    """
    this function creates a directory
    Args:
    path_to_dir (Path): path to the directory

    Returns:
    None

    """
    try:
        for path_to_dir in path_to_dirs:
            os.makedirs(path_to_dir,exist_ok=True,mode=777)
            if verbose:
                logging.info(f"directory {path_to_dir} created")
    except Exception as e:
        logging.info(e)
        raise e
    
@ensure_annotations
def write_json(path_to_json:Path,data:dict):
    """
    this function writes a json file
    Args:
    path_to_json (Path): path to the json file
    data (dict): data to be written to the json file

    Returns:
    None

    """
    try:
        with open(path_to_json, "w") as f:
            json.dump(data, f,indent=4)
            logging.info(f"json file {path_to_json} written")
    except Exception as e:
        logging.info(e)
        raise e
@ensure_annotations    
def load_json(path_to_json:Path)->dict:
    """
    this function loads a json file
    Args:
    path_to_json (Path): path to the json file

    Returns:
    dict: data loaded from the json file

    """
    try:
        with open(path_to_json, "r") as f:
            data = json.load(f)
            logging.info(f"json file {path_to_json} loaded")
            return ConfigBox(data)
    except Exception as e:
        logging.info(e)
        raise e
    
@ensure_annotations
def save_binary_file(path_to_binary_file:Path,data:Any):
    """
    this function saves a binary file
    Args:
    path_to_binary_file (Path): path to the binary file
    data (Any): data to be written to the binary file

    Returns:
    None

    """
    try:
        joblib.dump(data, f)
        logging.info(f"binary file {path_to_binary_file} written")
    except Exception as e:
        logging.info(e)
        raise e
    
@ensure_annotations
def load_binary_file(path_to_binary_file:Path)->bytes:
    """
    this function loads a binary file
    Args:
    path_to_binary_file (Path): path to the binary file

    Returns:
    bytes: data loaded from the binary file

    """
    try:
        data = joblib.load(f)
        logging.info(f"binary file {path_to_binary_file} loaded")
        return data
    except Exception as e:
        logging.info(e)
        raise e
    
@ensure_annotations
def get_size(path:Path)->str:
    """
    this function gets the size of a file
    Args:
    path (Path): path to the file

    Returns:
    str: size of the file in KB

    """
    try:
        size = round(os.path.getsize(path)/1024)
        return f"{size} KB"
    except Exception as e:
        logging.info(e)
        raise e
    
@ensure_annotations
def decode_base64(base64_string:str,file_name:str)->bytes:
    img_data = base64.b64decode(base64_string)
    with open(file_name, "wb") as f:
        f.write(img_data)
        f.close()

@ensure_annotations
def encode_base64(file_name:str)->str:
    with open(file_name, "rb") as f:
        img_data = f.read()
        f.close()
    return base64.b64encode(img_data)

def connect_to_s3():
    cridentials = read_yaml(Path(".aws/cridentials.yaml"))
    s3 = boto3.client(
    's3',
    aws_access_key_id=cridentials.ACCESS_KEY,
    aws_secret_access_key=cridentials.SECRET_KEY
    )
    return s3


def train_val_split(path_csv,path_image,test_size=0.2):
        
        data_labels= pd.read_csv(path_csv)
        data_labels=data_labels.assign(images_path=data_labels["images"].apply(lambda x:os.path.join(path_image, x)))
        data_labels.drop(columns=["images"], inplace=True)
        data_labels=data_labels.sample(frac=1).reset_index(drop=True)
        data_labels=data_labels.reset_index(drop=True)
        
        train_images,test_images,train_labels,test_labels=train_test_split(data_labels["images_path"],data_labels["label"],test_size=test_size,stratify=data_labels["label"])
        return train_images,test_images,train_labels,test_labels