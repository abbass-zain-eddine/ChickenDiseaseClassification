import os
from pathlib import Path
import logging 



logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')

project_name="CNNClassifier Project"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb"
    
]


for path in list_of_files:
    path=Path(path)
    dir,file=os.path.split(path)


    if dir !="":
        os.makedirs(dir,exist_ok=True)
        logging.info(f"creating directory: {dir} for the file {file}")

    if (not os.path.exists(path)) or (os.path.getsize(path)==0) :
        with open(path,"w") as file_obj:
            pass
        logging.info(f"Creating empty file: {path}")
    
    else:
        logging.info(f"{file} is already exist")

