from multiprocessing.spawn import import_main_path
import os
import yaml
import logging
import time
import pandas as pd
import json
import textwrap
from zipfile import ZipFile
from pathlib import Path
import pathlib

def get_appended_path(path_to_append):
    # Root Path
    base_project_folder = str(pathlib.Path().resolve())
    appended_path=base_project_folder +f"\\{path_to_append}"
    if os.path.isdir(appended_path):
        create_directories([appended_path])
    return appended_path

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)

def read_dataframe(config)->pd.DataFrame:
    '''
    Read the csv file and return the dataframe
    config : configuration file for the project.
    '''
    csv_path=os.path.join(get_appended_path(config["data"]["RAW_DIR"]),config["data"]["TRAIN_FILE"])
    train_df=pd.read_csv(csv_path)
    return train_df

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logging.info(f"created directory at: {path}")
        else:
            logging.info(f"{path} exists.")


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


def unzip_file(source:str,dest:str)->None:
    logging.info(f"Extraction started to {dest}")
    with ZipFile(source, "r") as zip_f:
        zip_f.extractall(dest)
    logging.info(f"Extracted {source} to {dest}")
