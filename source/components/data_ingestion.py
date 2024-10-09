import os 
import sys

import pandas as pd 
import numpy as np

from source.constants import *
from source.config import *

from dataclasses import dataclass
from source.logger import logging
from source.exception import CustomException
from source.components.data_transformation import DataTransformation
from source.components.data_transformation import DataTransformationConfig

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH
    raw_data_path:str = RAW_FILE_PATH

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def iniitiate_data_ingestion(self):
        try:
            df = pd.read_csv(DATASET_PATH)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok = True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index = False)

            train_set, test_set = train_test_split(df, test_size = 0.20, random_state = 42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok = True )
            train_set.to_csv(self.data_ingestion_config.train_data_path, header = True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok = True )
            test_set.to_csv(self.data_ingestion_config.test_data_path, header = True)

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path

            )


        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.iniitiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.inititate_data_transformation(train_data_path, test_data_path)


#df = pd.read_csv(os.path.join('C:\Users\...data\finaltrain.csv'))