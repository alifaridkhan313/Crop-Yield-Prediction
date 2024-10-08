import os 
import sys 
from dataclasses import dataclass

import xgboost
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV, 
    train_test_split
)

from sklearn.metrics import r2_score

from source.exception import CustomException
from source.logger import logging
from source.utilis import save_object, evaluate_models, model_metrics

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            X_train, y_train, X_test, y_test = (

                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]

            )
            
            logging.info("Model initiallizing done successfully")
            models = {
                "Linear Regressor" : LinearRegression(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "Gradient Boosting Regressor" : GradientBoostingRegressor(),
                "Support Vector Regressor" : SVR(),
                "XGBoost Regressor" : XGBRegressor(),
                "LightGBM Regressor" : (),
            }

            params={
                "Linear Regressor": {},

                "Decision Tree Regressor": {
                    "max_depth": [1,2,4,6,8,10],
                    "min_samples_split": [1,2,4,6,8,10],
                    "min_samples_leaf": [1,2,4,6,8,10],
                    "random_state": [0,42],
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "splitter": ['best','random'],
                    "max_features": ['sqrt','log2']
                },

                "Random Forest Regressor": {
                    "max_depth": [1,2,4,6,8,10],
                    "n_estimators": [10,50,100,200,300,400],
                    "min_samples_split": [1,2,4,6,8,10],
                    "min_samples_leaf": [1,2,4,6,8,10],
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "max_features": ['sqrt','log2']
                },

                "Gradient Boosting Regressor": {
                    "n_estimators": [8,16,32,64,128,100,200,300],
                    "learning_rate": [0.01,0.1,0.2,0.3],
                    "max_depth": [3,4,5],
                    "min_samples_split": [2,3,5],
                    "min_samples_leaf": [1,2,5],
                    "subsample": [0.8,0.9,1.0,0.6,0.7,0.75,0.8,0.85,0.9],
                    "max_features": ['auto', 'sqrt', 'log2'],
                    "loss": ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "criterion": ['squared_error', 'friedman_mse']
                },

                "Support Vector Regressor": {
                    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                    "C":  [0.1,1,5,7,10],
                    "degree": [1,2,3,4,5],
                    "epsilon":  [0.1,0.5,1,2,3],
                    "gamma": ['scale', 'auto'],
                    'depth': [6,8,10]
                },

                "XGBoost Regressor": {
                    "max_depth": [3,5,7,9],
                    "learning_rate": [0.01,0.1,0.2,0.5], 
                    "n_estimators": [10,50,100,200,300],
                    "gamma": [0,0.25,0.5,0.75,1], 
                    "subsample": [0.5,0.75,1],
                    "colsample_bytree": [0.5,0.75,1],
                    "reg_alpha": [0,0.5,1],
                    "reg_lambda": [0,0.5,1],
                    "min_child_weight": [1,5,10]
                },

                "LightGBM Regressor": {
                    "num_leaves": [2,4,6,8,10],
                    "max_depth": [3,5,7,9],
                    "learning_rate": [0.01,0.1,0.2,0.5],
                    "n_estimators": [10,50,100,200,300],
                    "min_child_samples": [1,5,10],
                    "reg_alpha": [0,0.5,1],
                    "reg_lambda": [0,0.5,1],
                    "min_data_in_leaf": [1,5,10],
                    "min_sum_hessian_in_leaf": [1e-3,1e-2,1e-1]
                }
                
            }

            logging.info("***Hyperparameter tuning done successfully***")
            model_report : dict = evaluate_models(
                                                  X_train = X_train, 
                                                  y_train = y_train, 
                                                  X_test = X_test, 
                                                  y_test = y_test, 
                                                  models = models, 
                                                  param = params
                                                  )
            
            #to get the best model r2 score from dict 
            best_model_score = max(sorted(model_report.values()))

            #to get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("***Unable to find best model***")
            
            logging.info(f"Best model has been found for both i.e; train & test dataset successfully")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            mae, rmse, r2 = model_metrics(y_test, predicted)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info('***Finally Model Training Completed Successfully***')
            
            return mae, rmse, r2 
        
        except Exception as e:
            logging.info('***Error occured while training the model***')
            raise CustomException(e, sys)