import os 
import sys
import pickle
import numpy as np

from source.logger import logging
from source.exception import CustomException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_obj(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def model_metrics(true, predicted):
    try :
        
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e, sys)