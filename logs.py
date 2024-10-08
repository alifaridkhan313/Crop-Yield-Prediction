import os 
import sys
from flask import Flask 
from source.logger import logging 
from source.exception import CustomException
app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def index():

    try: 
        raise Exception("We are just testing our Exception file")
    except Exception as e: 
        ML = CustomException(e, sys)
        logging.info(ML.error_message
                     )
    logging.info("testing purpose")

    return "hey its working" 


if __name__ == '__main__':
    app.run(debug = True)