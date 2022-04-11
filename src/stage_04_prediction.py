import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_test_dataframe, read_yaml, create_directories
import random
import pandas as pd
from src.utils import file_operations
from src.utils.preprocessing import *

STAGE = "Prediction" ## <<< change stage name

'''
Final dataframe needs to be in this format

'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
'No_spending', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP','Cabin_deck', 'Cabin_side'

'''

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def create_dataframe(age,room_service,food_court,shopping_mall,spa,vrdeck,no_spending,home_planet,cryosleep,destination,vip,cabin_deck,cabin_side):
    details = {
    'Age' : [age],
    'RoomService' : [room_service],
    'FoodCourt' : [food_court],
    'ShoppingMall' : [shopping_mall],
    'Spa' : [spa],
    'VRDeck' : [vrdeck],
    'No_spending' : [no_spending],
    'HomePlanet' : [home_planet],
    'CryoSleep':[cryosleep],
    'Destination':[destination],
    'VIP':[vip],
    'Cabin_deck':[cabin_deck],
    'Cabin_side':[cabin_side]
    }
    features=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','No_spending', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP','Cabin_deck', 'Cabin_side']
    df=pd.DataFrame(details,columns= features)


def preprocess_data(config):
    test_df=read_test_dataframe(config)
    test_df=create_age_group_features(test_df)
    test_df=extract_cabin_info(test_df)
    numeric_data,categorical_data=split_to_numeric_cat_data(test_df)
    print(numeric_data.columns)
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
    params = read_yaml(params_path)
    file_op=file_operations.FileOperations(config['artifacts'])
    loaded_model=file_op.load_model('LightGBM')
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
