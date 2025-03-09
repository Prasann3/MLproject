import os , sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.execption import CustomException
from src.util import save_object , evaluate_models
@dataclass
class ModelTrainerConfig :
    trained_model = os.path.join('artifacts' , 'trained_model.pkl');

class ModelTrainer :
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig();
    
    def initiate_model_training(self,train_arr , test_arr , preprocessor_path) :

        logging.info("Retriving input and output data")
        x_train , y_train , x_test , y_test = (
            train_arr[: , :-1] ,
            train_arr[: , -1],
            test_arr[: , : - 1] ,
            test_arr[: , -1]
        )
        models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
        
        report : dict = evaluate_models(x_train , y_train , x_test , y_test , models)
        max_score = max(report.values())
        if max_score < 0.6 :
            raise CustomException("No Best Model Found")
        
        
        for name in report.keys() :
            if report[name] == max_score :
                best_model_name = name
                break;
        
        best_model = models[best_model_name]
        save_object(
            self.trained_model_config.trained_model ,
            best_model
        )
        logging.info("Best model found")
        return max_score;
