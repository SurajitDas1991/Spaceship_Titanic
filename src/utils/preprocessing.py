from tqdm import tqdm
import logging
from src.utils.common import *
from src.utils import preprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import CountFrequencyEncoder
import pickle

def encode_categorical_data(categorical_data):
    encoder = CountFrequencyEncoder(encoding_method='frequency')
    categorical_data=encoder.fit_transform(categorical_data)
    return categorical_data

def impute_categorical_data(categorical_data):
    imputer = CategoricalImputer()
    CategoricalImputer(fill_value='frequent',return_object=False)
    categorical_data=imputer.fit_transform(categorical_data)
    return categorical_data

def scale_test_numeric_features(X_imputed):
    scaler = pickle.load(open("./scaler.pkl",'rb'))
    numeric_data_scaled=scaler.fit_transform(X_imputed)
    numeric_data_imputed_df=pd.DataFrame(numeric_data_scaled,columns=X_imputed.columns)
    return numeric_data_imputed_df

def scale_numeric_features(X_imputed):

    scaler = preprocessing.MaxAbsScaler()
    numeric_data_scaled=scaler.fit_transform(X_imputed)
    numeric_data_imputed_df=pd.DataFrame(numeric_data_scaled,columns=X_imputed.columns)
    pickle.dump(scaler, open("./scaler.pkl",'wb'))
    return numeric_data_imputed_df


def impute_continous_features(numeric_data):
    median_imputer = MeanMedianImputer(imputation_method='median')
    # fit the imputer
    X_imputed=median_imputer.fit_transform(numeric_data)
    return X_imputed


def log_transform(train_df,cols):
    for col in cols:
        train_df[col]=np.log(1+train_df[col])
    return train_df

def features_to_be_dropped(train_df,columns):
    train_df.drop(columns, axis=1, inplace=True)
    return train_df


def split_to_numeric_cat_data(train_df):
    numeric_data = train_df.select_dtypes(include=[np.number])
    categorical_data = train_df.select_dtypes(exclude=[np.number])
    return numeric_data,categorical_data

def extract_cabin_info(train_df):
    train_df['Cabin'].fillna("Z/9999/Z",inplace=True)
    train_df['Cabin_deck']=train_df['Cabin'].apply(lambda x:x.split('/')[0])
    train_df['Cabin_number']=train_df['Cabin'].apply(lambda x:x.split('/')[1]).astype(int)
    train_df['Cabin_side']=train_df['Cabin'].apply(lambda x:x.split('/')[2])
    train_df.loc[train_df['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
    train_df.loc[train_df['Cabin_number']==9999,'Cabin_number']=np.nan
    train_df.loc[train_df['Cabin_side']=='Z', 'Cabin_side']=np.nan
    train_df.drop('Cabin',axis=1,inplace=True)
    return train_df




def create_expenditure_feature(train_df,numeric_data)->pd.DataFrame:
    exp_feats=numeric_data[1:].columns
    train_df['Expenditure']=train_df[exp_feats].sum(axis=1)
    train_df['No_spending']=(train_df['Expenditure']==0).astype(int)
    return train_df

def create_age_group_features(train_df):
    train_df['Age_group']=np.nan
    train_df.loc[train_df['Age']<=12,'Age_group']='Age_0-12'
    train_df.loc[(train_df['Age']>12) & (train_df['Age']<18),'Age_group']='Age_13-17'
    train_df.loc[(train_df['Age']>=18) & (train_df['Age']<=25),'Age_group']='Age_18-25'
    train_df.loc[(train_df['Age']>25) & (train_df['Age']<=30),'Age_group']='Age_26-30'
    train_df.loc[(train_df['Age']>30) & (train_df['Age']<=50),'Age_group']='Age_31-50'
    train_df.loc[train_df['Age']>50,'Age_group']='Age_51+'
    return train_df
