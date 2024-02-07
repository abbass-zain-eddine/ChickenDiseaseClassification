import os
import urllib.request as request
import zipfile
from CNNClassifierProject.entity.logger import logging
from CNNClassifierProject.utils.common import get_size,connect_to_s3
from CNNClassifierProject.entity.config_entity import DataIngestionConfig
from pathlib import Path
import requests as req
import boto3

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """Download the data"""
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                 filename= self.config.local_data_file
            )
            logging.info(f"Downloaded {filename} to {self.config.local_data_file}")
        else:
            logging.info(f"{self.config.local_data_file} already exists of size {get_size(Path(self.config.local_data_file))}")

    def download_data_s3(self):
        """Download the data from s3 bucket"""
        if not os.path.exists(self.config.local_data_file):

            s3=connect_to_s3()
            s3.download_file(self.config.source_URL.split("/")[2],self.config.source_URL.split("/")[3],self.config.local_data_file)
            
            logging.info(f"Downloaded {self.config.source_URL.split('/')[3]} to {self.config.local_data_file}")
        else:
            logging.info(f"{self.config.local_data_file} already exists of size {get_size(Path(self.config.local_data_file))}")


    def unzip_data(self):
        """Unzip the data"""
        if not os.path.exists(self.config.unzip_dir):
            os.makedirs(self.config.unzip_dir,exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(self.config.unzip_dir)
            logging.info(f"Unzipped {self.config.local_data_file} to {self.config.unzip_dir}")
        else:
            logging.info(f"{self.config.unzip_dir} already exists of size {get_size(Path(self.config.unzip_dir))}")