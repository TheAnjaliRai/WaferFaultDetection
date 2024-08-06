# import sys,os

# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
# from src.exception import CustomException






# class TraininingPipeline:

    
#     def start_data_ingestion(self):
#         try:
#             data_ingestion = DataIngestion()
#             feature_store_file_path = data_ingestion.initiate_data_ingestion()
#             return feature_store_file_path
            
#         except Exception as e:
#             raise CustomException(e,sys)
        


        
    
#     def start_data_transformation(self, feature_store_file_path):
#         try:
#             data_transformation = DataTransformation(feature_store_file_path= feature_store_file_path)
#             train_arr, test_arr,preprocessor_path = data_transformation.initiate_data_transformation()
#             return train_arr, test_arr,preprocessor_path
            
#         except Exception as e:
#             raise CustomException(e,sys)


#     def start_model_training(self, train_arr, test_arr):
#         try:
#             model_trainer = ModelTrainer()
#             model_score = model_trainer.initiate_model_trainer(
#                 train_arr, test_arr
#             )
#             return model_score
            
#         except Exception as e:
#             raise CustomException(e,sys)
        

#     def run_pipeline(self):
#         try:
#             feature_store_file_path = self.start_data_ingestion()
#             train_arr, test_arr,preprocessor_path = self.start_data_transformation(feature_store_file_path)
#             r2_square = self.start_model_training(train_arr, test_arr)
            
            
#             print("training completed. Trained model score : ", r2_square)

#         except Exception as e:
#             raise CustomException(e, sys)


import sys
import os
import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TraininingPipeline:

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Feature store file path: {feature_store_file_path}")
            return feature_store_file_path
            
        except Exception as e:
            logging.error("Error during data ingestion")
            raise CustomException(e, sys)
        
    def start_data_transformation(self, feature_store_file_path):
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            return train_arr, test_arr, preprocessor_path
            
        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):
        try:
            logging.info("Starting model training")
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed. Model score: {model_score}")
            return model_score
            
        except Exception as e:
            logging.error("Error during model training")
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try:
            logging.info("Started training pipeline")
            feature_store_file_path = self.start_data_ingestion()
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            r2_square = self.start_model_training(train_arr, test_arr)
            logging.info(f"Training completed. Trained model score: {r2_square}")
            return "Training Completed."

        except Exception as e:
            logging.error("Error during training pipeline execution")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

