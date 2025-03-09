import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.execption import CustomException
from src.logger import logging
from src.util import get_num_and_cat_features , save_object
import os , pickle

@dataclass
class DataTransformationConfig : 
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')


class DataTransformation :

    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig();

    def get_data_transformer_obj(self) :

       try : 
        num_features , cat_features = get_num_and_cat_features()

        num_pipeline = Pipeline(
            steps=[
                ('Imputer' , SimpleImputer(strategy='median')) ,
                ('Scaler' , StandardScaler())
            ]
        )
        cat_pipeline = Pipeline(
            steps=[ 
                ('Imputer' , SimpleImputer(strategy='most_frequent')) ,
                ('Encoder' , OneHotEncoder())
            ]
        )
        logging.info("Catagorical and numerical features logging completed")

        preprocessor = ColumnTransformer(
            [
                ('numpipeline' , num_pipeline , num_features) ,
                ('catpipeline' , cat_pipeline , cat_features)
            ]
        )
        return preprocessor
       except Exception as e :
          raise CustomException(e , sys);

    def initiate_data_transformation(self , train_path , test_path) :
       
       try :
          
          df_train = pd.read_csv(train_path)
          df_test = pd.read_csv(test_path)
          logging.info("Reading of train and test data is completed")
          logging.info("Getting preprocessor obejct")
          preprocessor = self.get_data_transformer_obj();
          target_coloumn = 'math_score'
          input_feature_train = df_train.drop(target_coloumn , axis=1)
          input_feature_test =  df_test.drop(target_coloumn , axis=1)
          target_coloumn_train = df_train[target_coloumn]
          target_coloumn_test = df_test[target_coloumn]
          input_feature_train_arr = preprocessor.fit_transform(input_feature_train)
          input_feature_test_arr = preprocessor.transform(input_feature_test)
          train_arr = np.c_[
             input_feature_train_arr , np.array(target_coloumn_train)
          ]
          test_arr = np.c_[
             input_feature_test_arr , np.array(target_coloumn_test)
          ]

          save_object(
             self.data_transformation_config.preprocessor_obj_file_path,
             preprocessor
          )
          logging.info("Preprocessor saved in the memory")

          return (
             train_arr ,
             test_arr,
             self.data_transformation_config.preprocessor_obj_file_path
          )

       except Exception as e:
          raise CustomException(e , sys);

          
       




if __name__ == '__main__' :
   obj = DataTransformation().get_data_transformer_obj();
   print(obj)