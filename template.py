import os 
import sys
import logging

from pathlib import Path 

#a simple while loop logic to ask for input name of folder
while True: 
    folder_name = input("Enter name for your folder:  ")
    if folder_name != "":
        break


#creating list of files under that folder 
#usinhg f string
#__init__.py is a constructor file

list_of_files = [

    f"{folder_name}/__init__.py", 
    f"{folder_name}/components/__init__.py",
    f"{folder_name}/config/__init__.py",
    f"{folder_name}/consraints/__init__.py",
    f"{folder_name}/entity/__init__.py",
    f"{folder_name}/exception/__init__.py",
    f"{folder_name}/logger/__init__.py",
    f"{folder_name}/pipeline/__init__.py",
    f"config/config.yaml",
    "schema.yaml", 
    "app.py", 
    "main.py", 
    "logs.py", 
    "exception.py", 
    "setup.py", 
    "requirements.txt",
    

]

#simple for loop logic to  create and split each file
#with their respective directorties
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,  filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok = True)
        logging.info(f"Created directry: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f: 
            pass
            logging.info(f"Created an empty file: {filepath}")

    else: 
        logging.info(f"File is already present at: {filepath}")
        #logging.info(f"File {filepath} already exists.")