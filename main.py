from Networksecurity.components.data_ingestion import DataIngestion
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from Networksecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys)