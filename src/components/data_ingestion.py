import os , sys
from src.logger import logging
from src.execption import CustomException
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation , DataTransformationConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig :
    train_data_path:str = os.path.join('artifacts' , 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv');

class DataIngestion :

    def __init__(self) :
          self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) :
         logging.info("Entered the data ingestion component")

         try :
              df = pd.read_csv('notebook\data\stud.csv');
              logging.info("Read the dataset as dataframe")

              os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok=True)
              df.to_csv(self.ingestion_config.raw_data_path , index=False,header=True)
              logging.info("Raw data saved as csv")
              logging.info("Train Test Split Initiated")
              train_data , test_data = train_test_split(df , test_size=0.2 , random_state=69)
              train_data.to_csv(self.ingestion_config.train_data_path , index=False,header=True)
              test_data.to_csv(self.ingestion_config.test_data_path , index=False,header=True)
              logging.info("Ingestion Completed")

              return (
                   self.ingestion_config.train_data_path ,
                   self.ingestion_config.test_data_path
              )
         except Exception as e :
              raise CustomException(e , sys);
              

if __name__ == "__main__" :
     data_ingester = DataIngestion()
     train_data , test_data = data_ingester.initiate_data_ingestion()   
      
     data_transformer = DataTransformation()
     train_array , test_array , pre = data_transformer.initiate_data_transformation(train_data , test_data)
     trainer = ModelTrainer()
     print(trainer.initiate_model_training(train_array , test_array , pre))


      
    

     