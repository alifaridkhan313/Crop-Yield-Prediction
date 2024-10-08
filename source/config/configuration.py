import os 
import sys

from source.constraints import *

ROOT_DIR = ROOT_DIR_KEY

"""
In this file we will be making some directries 
here are the list of them: 

1. dataset path 
2. raw file path 
3. train file path
4. test file path 
5. preprocessing object file path
6. train transform file path
7. test transform file path
8. feature enginnering object file path

This will make our work easy and we can easily find the path of the files we need to use.

"""


###dataset related paths:

#dataset path
DATASET_PATH = os.path.join(
                            ROOT_DIR, 
                            DATA_DIR, 
                            DATA_DIR_KEY
                            )


#raw dataset file path
RAW_FILE_PATH = os.path.join(
                             ROOT_DIR, 
                             ARTIFACT_DIR_KEY, 
                             DATA_INGESTION_KEY,  
                             DATA_INGESTION_RAW_DATA_DIR, 
                             RAW_DATA_DIR_KEY
                             )


#train dataset file path
TRAIN_FILE_PATH = os.path.join(
                               ROOT_DIR, 
                               ARTIFACT_DIR_KEY, 
                               DATA_INGESTION_KEY, 
                               DATA_INGESTION_INGESTED_DATA_DIR_KEY, 
                               TRAIN_DATA_DIR_KEY
                               )


#test dataset file path
TEST_FILE_PATH = os.path.join(
                              ROOT_DIR, 
                              ARTIFACT_DIR_KEY, 
                              DATA_INGESTION_KEY, 
                              DATA_INGESTION_INGESTED_DATA_DIR_KEY, 
                              TEST_DATA_DIR_KEY
                              )

#=====================End of dataset related paths=====================


#data transformation related paths:

#preprocessing object file path
PREPROCESING_OBJECT_FILE = os.path.join(
                                     ROOT_DIR, 
                                     ARTIFACT_DIR_KEY, 
                                     DATA_TRANSFORMATION_ARTIFACT, 
                                     DATA_PREPROCESSED_DIR, 
                                     DATA_TRANSFORMTION_PROCESSING_OBJECT
                                     )


#transform train file path
TRANSFORM_TRAIN_FILE_PATH = os.path.join(
                                         ROOT_DIR, 
                                         ARTIFACT_DIR_KEY, 
                                         DATA_TRANSFORMATION_ARTIFACT,
                                         DATA_TRANSFORM_DIR, 
                                         TRANSFORM_TRAIN_DIR_KEY
                                         )


#transform test file path
TRANSFORM_TEST_FILE_PATH = os.path.join(
                                        ROOT_DIR, 
                                        ARTIFACT_DIR_KEY, 
                                        DATA_TRANSFORMATION_ARTIFACT,
                                        DATA_TRANSFORM_DIR, 
                                        TRANSFORM_TEST_DIR_KEY
                                        )


#feature enginnering object file path
FEATURE_ENGINNERING_OBJECT_FILE_PATH = os.path.join(
                                                    ROOT_DIR, 
                                                    ARTIFACT_DIR_KEY, 
                                                    DATA_TRANSFORMATION_ARTIFACT,
                                                    DATA_PREPROCESSED_DIR,' feature_enginnering.pkl'
                                                    )