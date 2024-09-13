import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging
from utils import evaluate_models, save_object


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "Catboost": CatBoostRegressor(verbose = 0),
                "Adaboost":AdaBoostRegressor()
            }

            params = {
                "Decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    "max_features": ['sqrt', 'log2']
                },
                "Random Forest": {
                    'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    "max_features": ['sqrt', 'log2'],
                    "n_estimators":[8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting":{
                    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate':[0.1, 0.01, 0.05, 0.0001],
                    'subsample':[0.6, 0.7, 0.75, 0.8, 0.9],
                    'criterion':['friedman_mse', 'squared_error'],
                    "max_features": ['sqrt', 'log2'],
                    "n_estimators":[8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K-Neighbours": {
                    "weights": ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBoost": {
                    'learning_rate':[0.1, 0.01, 0.05, 0.0001],
                    "n_estimators":[8, 16, 32, 64, 128, 256]
                },
                "Catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)

            # to get the best model from dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            logging.info(f"Best model is {best_model_name}")

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both the training and test data")

            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                                      obj = best_model)
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2
        except Exception as e:
            raise CustomException(e, sys)
