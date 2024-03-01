from CNNClassifierProject.config.configuration import ConfigurationManager
from CNNClassifierProject.components.data_ingestion import DataIngestion
from CNNClassifierProject.entity.logger import logging


STAGE_NAME = 'data_ingestion'

class DataIngestionTrainingPipline:
    def __init__(self):
        pass

    def main(self):
        try:
            config=ConfigurationManager()
            data_ingestion_config= config.get_data_ingestion_config()
            data_ingestion= DataIngestion(config=data_ingestion_config)
            data_ingestion.download_data_s3()
            data_ingestion.unzip_data()
        except Exception as e:
            logging.info(e)
            raise e
        

if __name__ == '__main__':
    try:
        logging.info(f" Stage Name: {STAGE_NAME} start")
        obj= DataIngestionTrainingPipline()
        obj.main()
        logging.info(f" Stage Name: {STAGE_NAME} end")
    except Exception as e:
        logging.info(e)
        raise e