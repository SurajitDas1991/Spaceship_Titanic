import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_test_dataframe, read_yaml
import pandas as pd
from src.utils import file_operations
from src.utils.preprocessing import *

STAGE = "Prediction" ## <<< change stage name



logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )




def preprocess_data(config):
    test_df=read_test_dataframe(config)
    test_df=create_age_group_features(test_df)
    test_df=extract_cabin_info(test_df)
    numeric_data,categorical_data=split_to_numeric_cat_data(test_df)

    test_df=create_expenditure_feature(test_df,numeric_data)
    cols_to_drop=['PassengerId','Age_group', 'Cabin_number','Expenditure','Name']
    test_df=features_to_be_dropped(test_df,cols_to_drop)

    #print(train_df.columns)
    X=test_df.copy()
    numeric_data,categorical_data=split_to_numeric_cat_data(X)
    columns_for_transformation=[numeric_data.columns[1:]]
    #print(columns_for_transformation)
    #Take care of missing values , skewness
    #Numerical features
    X=log_transform(X,columns_for_transformation)
    numeric_data,categorical_data=split_to_numeric_cat_data(X)
    numeric_data=impute_continous_features(numeric_data)
    numeric_data=scale_test_numeric_features(numeric_data)
    #Categorical features
    categorical_data=impute_categorical_data(categorical_data)
    categorical_data=encode_categorical_data(categorical_data)
    final_df=pd.concat([numeric_data,categorical_data], axis=1)
    return final_df

def predict(df,loaded_model):
    result=loaded_model.predict(df)
    print(result)

def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    file_op=file_operations.FileOperations(config['artifacts']['TRAINED_MODEL_DIR'])
    loaded_model=file_op.load_model('StackingClassifier')
    df=preprocess_data(config)
    if loaded_model is not None:
        predict(df,loaded_model)



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
