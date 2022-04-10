import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import get_appended_path, read_yaml, create_directories
import random
import pandas as pd

STAGE = "Get data" ## <<< change stage name

train_df=pd.DataFrame()

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def read_dataframe(config):
    csv_path=os.path.join(get_appended_path(config["data"]["RAW_DIR"]),config["data"]["TRAIN_FILE"])
    train_df=pd.read_csv(csv_path)
    return train_df

def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    train_df=read_dataframe(config)
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
