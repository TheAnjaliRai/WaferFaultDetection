# 









import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join('artifacts')
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }

    def evaluate_models(self, X, y, models):
        try:
            logging.info("Starting model evaluation")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model_name = list(models.keys())[i]
                logging.info(f"Training model: {model_name}")
                model.fit(X_train, y_train)  # Train model

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[model_name] = test_model_score

                logging.info(f"{model_name} - Training accuracy: {train_model_score}, Test accuracy: {test_model_score}")

            logging.info("Completed model evaluation")
            return report

        except Exception as e:
            logging.error(f"Error in evaluate_models: {e}")
            raise CustomException(e, sys)

    def get_best_model(self, x_train, y_train, x_test, y_test):
        try:
            logging.info("Getting the best model")
            model_report = self.evaluate_models(
                X=x_train, 
                y=y_train, 
                models=self.models
            )

            logging.info(f"Model report: {model_report}")
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            logging.error(f"Error in get_best_model: {e}")
            raise CustomException(e, sys)
        
    def finetune_best_model(self, best_model_object, best_model_name, X_train, y_train):
        try:
            logging.info(f"Finetuning the best model: {best_model_name}")
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1 )
            
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            logging.info(f"Best parameters for {best_model_name}: {best_params}")

            finetuned_model = best_model_object.set_params(**best_params)

            logging.info(f"Finetuned model: {best_model_name}")
            return finetuned_model
        
        except Exception as e:
            logging.error(f"Error in finetune_best_model: {e}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target feature")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Extracting model config file path")

            model_report = self.evaluate_models(X=x_train, y=y_train, models=self.models)

            logging.info(f"Model report: {model_report}")
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = self.models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )

            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
            logging.info(f"Best model name {best_model_name} and score: {best_model_score}")

            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.5")
            
            logging.info("Best model found on both training and testing dataset")

            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            return self.model_trainer_config.trained_model_path

        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {e}")
            raise CustomException(e, sys)
