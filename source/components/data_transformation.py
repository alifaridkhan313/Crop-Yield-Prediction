import os
import sys

import numpy as np
import  pandas as pd

from source.constants import *
from source.logger import logging
from source.exception import CustomException
from source.utilis import save_obj
from source.config.configuration import PREPROCESING_OBJECT_FILE, TRANSFORM_TRAIN_FILE_PATH, TRANSFORM_TEST_FILE_PATH, FEATURE_ENGINNERING_OBJECT_FILE_PATH
from source.config.configuration import *

from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

"""
if we have missing values we will be using
from sklearn.impute import SimpleImputer
"""



class Feature_Engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info("========================Feature Engineering started successfully========================")


    def add_ratios_and_proportions(self, df):
        #ratios and proportions
        try:
            df['Yield_per_Area'] = df['Production'] / df['Area']
            df['Fertilizer_Intensity'] = df['Fertilizer'] / df['Area']
            df['Pesticide_Intensity'] = df['Pesticide'] / df['Area']

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def add_interaction_terms(self, df):
        #interaction terms
        try:
            df['Area_Fertilizer'] = df['Area'] * df['Fertilizer']
            df['Area_Pesticide'] = df['Area'] * df['Pesticide']
            df['Production_Fertilizer'] = df['Production'] * df['Fertilizer']

            return df

        except Exception as e:
            raise CustomException(e, sys)
    
    logging.info("***features created successfully***")

@dataclass 
class DataTransformationConfig():
    proccessed_object_file_path = PREPROCESING_OBJECT_FILE
    transform_train_path = TRANSFORM_TRAIN_FILE_PATH
    transform_test_path = TRANSFORM_TEST_FILE_PATH
    feature_enginnering_object_path = FEATURE_ENGINNERING_OBJECT_FILE_PATH


class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):

        try:
            categorical_columns = ['Crop','Season','State']
            numerical_column = [
                              'Crop_Year','Production','Area',
                              'Annual_Rainfall','Fertilizer', 
                              'Pesticide', 'Yield', 'Yield_per_Area',
                              'Fertilizer_Intensity', 'Pesticide_Intensity', 
                              'Area_Fertilizer', 'Area_Pesticide', 'Production_Fertilizer'
                              ]

            #numerical pipeline
            numerical_pipeline = Pipeline(steps = [
                ('Standard Scaler', StandardScaler(with_mean = False))
            ])

            #categorical Pipeline
            categorical_pipeline = Pipeline(steps = [
                ('One Hot Encoding', OneHotEncoder(handle_unknown = 'ignore')),
                ('Standard Scaler', StandardScaler(with_mean = False))
            ])

            #preprocessing
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline,numerical_column),
                ('categorical_pipeline', categorical_pipeline,categorical_columns),
            ])

            logging.info("***Pipeline Step Completed Successfully***")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def get_feature_engineering_object(self):
        try:
            #feature engineering pipeline
            feature_engineering = Pipeline(steps = [("Feature Engineering", Feature_Engineering())])
            return feature_engineering

        except Exception as e:
            raise CustomException(e, sys)
        
    def inititate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("***Obtaining feature engineering steps object***")
            feature_engineering_object = self.get_feature_engineering_object()

            train_df = feature_engineering_object.fit_transform(train_df)

            test_df = feature_engineering_object.transform(test_df)

            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")

            processing_obj = self.get_data_transformation_obj()

            target_column_name = "Yield"

            X_train = train_df.drop(columns = target_column_name, axis = 1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns = target_column_name, axis = 1)
            y_test = test_df[target_column_name]

            X_train = processing_obj.fit_transform(X_train)
            X_test = processing_obj.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_path), exist_ok = True)
            df_train.to_csv(self.data_transformation_config.transform_train_path, index = False, header = True)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_path), exist_ok = True)
            df_test.to_csv(self.data_transformation_config.transform_test_path, index = False, header = True)


            save_obj(file_path = self.data_transformation_config.proccessed_object_file_path,
                     obj = feature_engineering_object)

            save_obj(file_path = self.data_transformation_config.feature_enginnering_object_path,
                     obj = feature_engineering_object)
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.proccessed_object_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)