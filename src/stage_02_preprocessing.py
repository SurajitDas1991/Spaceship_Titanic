import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import *
from src.utils.preprocessing import *
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import CountFrequencyEncoder

STAGE = "Preprocess data" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )




def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    train_df=read_dataframe(config)
    train_df=create_age_group_features(train_df)
    train_df=extract_cabin_info(train_df)
    numeric_data,categorical_data=split_to_numeric_cat_data(train_df)
    print(numeric_data.columns)
    train_df=create_expenditure_feature(train_df,numeric_data)
    cols_to_drop=['PassengerId','Age_group', 'Cabin_number','Expenditure','Name']
    train_df=features_to_be_dropped(train_df,cols_to_drop)

    #print(train_df.columns)
    y=train_df['Transported'].copy().astype(int)
    X=train_df.drop('Transported',axis=1).copy()
    numeric_data,categorical_data=split_to_numeric_cat_data(X)
    columns_for_transformation=[numeric_data.columns[1:]]
    #print(columns_for_transformation)
    #Take care of missing values , skewness
    #Numerical features
    X=log_transform(X,columns_for_transformation)
    numeric_data,categorical_data=split_to_numeric_cat_data(X)
    numeric_data=impute_continous_features(numeric_data)
    numeric_data=scale_numeric_features(numeric_data)
    #Categorical features
    categorical_data=impute_categorical_data(categorical_data)
    categorical_data=encode_categorical_data(categorical_data)
    final_df=pd.concat([numeric_data,categorical_data], axis=1)
    #print(final_df)
    final_df.to_pickle("./final_df.pkl")
    y.to_pickle("./target.pkl")
    print(final_df.columns)

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
