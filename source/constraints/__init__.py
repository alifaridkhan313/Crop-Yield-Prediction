import os
import sys

from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR_KEY = os.getcwd()
DATA_DIR = "data"
DATA_DIR_KEY = "crop_yield.csv"

#.../DATA_DIR/DATA_DIR_KEY

ARTIFACT_DIR_KEY = "artifacts"

#data ingestion related variable
DATA_INGESTION_KEY = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR = "raw_data_dir"
DATA_INGESTION_INGESTED_DATA_DIR_KEY = "ingested_data_dir"
RAW_DATA_DIR_KEY = "raw_data.csv"
TRAIN_DATA_DIR_KEY = "train_data.csv"
TEST_DATA_DIR_KEY = "test_data.csv"


#data Transformation related variable
DATA_TRANSFORMATION_ARTIFACT = "data_transformation"
DATA_PREPROCESSED_DIR = "procceor"
DATA_TRANSFORMTION_PROCESSING_OBJECT = "processor.pkl"
DATA_TRANSFORM_DIR = "transformation"
TRANSFORM_TRAIN_DIR_KEY = "train_data.csv"
TRANSFORM_TEST_DIR_KEY = "test_data.csv"