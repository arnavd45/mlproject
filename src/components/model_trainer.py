import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            models = {
                 "Random Forest": RandomForestRegressor(),
                 "Adaboost Regressor": AdaBoostRegressor(),
                 "Linear Regression": LinearRegression(),
                 "Gradient Boosting": GradientBoostingRegressor(),
                 "KNN": KNeighborsRegressor(),
                 "Catboost Regressor": CatBoostRegressor(verbose=0),
                 "Xgboost": XGBRegressor(verbosity=0),
                 "SVM Regressor": SVR(),
                 "Decision Tree": DecisionTreeRegressor()
                }

            params={
                "Decision Tree":{
                    "criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "splitter":['best', 'random'],
                    "max_features":['sqrt', 'log2']
                },
                "Random Forest":{
                    "criterion":['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    "max_features":['sqrt', 'log2'],
                    "n_estimators":[100,200,300,400]
                },
                "Adaboost Regressor":{
                    "n_estimators":[100,200,300,400],
                    "learning_rate":[1,0.1,0.01,0.001],
                    "loss":['linear', 'square', 'exponential']
                },
                "Gradient Boosting":{
                    "loss":['squared_error', 'absolute_error', 'huber', 'quantile'],
                    "n_estimators":[100,200,300,400],
                    "subsample": [0.6, 0.7, 0.8, 0.9]
                },
                "Catboost Regressor":{
                    "depth":[6,8,10],
                    "learning_rate":[.1,.01,.05,.001],
                    "n_estimators":[100,200,300,400]

                },
                "Linear Regression":{},
                "KNN":{
                    "n_neighbors":[5,7,9,11],
                    "weights":['uniform', 'distance'],
                    "algorithm":['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "SVM Regressor":{
                    "gamma": ['scale', 'auto']
                },
                
                "Xgboost": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
                }

            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)


            best_model_score=max(model_report.values())

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f'Best found model on both training and testing dataset')


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)
            return best_model_score


        except Exception as e:
            raise CustomException(e,sys)
        
        
            

