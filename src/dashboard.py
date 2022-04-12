import streamlit as st
import argparse
import os
import pandas as pd
from src.utils.preprocessing import *
from src.utils import file_operations

original_df=pd.DataFrame()

def predict(df,loaded_model):
    result=loaded_model.predict(df)
    if result[0]==0:
        st.write(f"Unfortunately {original_df['Name'][0]} has not reached destination")
    else:
        st.write(f"{original_df['Name'][0]} has safely reached destination !!")


def create_dataframe(PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name
):
    details = {
    'PassengerId':[PassengerId],
    'HomePlanet': [HomePlanet],
    'CryoSleep':[CryoSleep],
    'Cabin':[Cabin],
    'Destination':[Destination],
    'Age': [Age],
    'VIP':[VIP],
    'RoomService': [RoomService],
    'FoodCourt': [FoodCourt],
    'ShoppingMall': [ShoppingMall],
    'Spa': [Spa],
    'VRDeck': [VRDeck],
    'Name': [Name]
    }
    features=['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','Age', 'VIP','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Name']
    df=pd.DataFrame(details,columns= features)
    global original_df
    original_df=df.copy()
    st.dataframe(df)
    return df

def preprocess_data(test_df):
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


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()
    config_path=parsed_args.config
    config = read_yaml(config_path)

    st.header("Did your near and dear one get transported ?")
    passenger_id=st.text_input("Enter passenger_id (ex - 0001_01):")
    home_planet=st.text_input("Enter home planet :")
    cryosleep=st.selectbox("CryoSleep",["TRUE","FALSE"])
    cabin=st.text_input("Enter cabin number (ex - B/0/P:")
    destination=st.text_input("Enter destination :")
    age=st.text_input("Enter age :")
    vip=st.selectbox("VIP",["TRUE","FALSE"])
    roomservice=st.text_input("Enter room service expense :")
    foodcourt=st.text_input("Enter food court expense :")
    shopping_mall=st.text_input("Enter shopping mall expense :")
    spa=st.text_input("Enter spa expense :")
    vrdeck=st.text_input("Enter vrdeck expense :")
    name=st.text_input("Enter name :")
    if st.button("PREDICT"):
        df=create_dataframe(passenger_id,home_planet,cryosleep,cabin,destination,age,vip,roomservice,foodcourt,shopping_mall,spa,vrdeck,name)
        df['Age']=pd.to_numeric(df['Age'],errors='coerce')
        file_op=file_operations.FileOperations(config['artifacts']['TRAINED_MODEL_DIR'])
        loaded_model=file_op.load_model('StackingClassifier')
        df=preprocess_data(df)
        if loaded_model is not None:
            predict(df,loaded_model)
