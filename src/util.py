import pandas as pd
import numpy as np
import os , sys , dill
from src.execption import CustomException
def get_num_and_cat_features() :
    df = pd.read_csv('artifacts\data.csv')
    df = df.drop('math_score' , axis=1)
    num_features = []
    cat_features = []
    features = list(df.columns)
    for feature in features :
        if df[feature].dtype == 'O' :
            cat_features.append(feature)
        else :
            num_features.append(feature)

    return num_features , cat_features;           

def save_object(file_path , obj) : 
    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)
        with open(file_path , 'wb') as file_obj :
            dill.dump(obj , file_obj);
    except Exception as e :
        raise CustomException(e , sys);