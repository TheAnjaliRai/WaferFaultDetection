# import sys
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler, FunctionTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import  StandardScaler

# from src.constant import *
# from src.exception import CustomException
# from src.logger import logging
# from src.utils.main_utils import MainUtils
# from dataclasses import dataclass

# @dataclass
# class DataTransformationConfig:
#     artifact_dir=os.path.join(artifact_folder)
#     transformed_train_file_path=os.path.join(artifact_dir, 'train.npy')
#     transformed_test_file_path=os.path.join(artifact_dir, 'test.npy') 
#     transformed_object_file_path=os.path.join( artifact_dir, 'preprocessor.pkl' )






# class DataTransformation:
#     def __init__(self,
#                  feature_store_file_path):
       
#         self.feature_store_file_path = feature_store_file_path

#         self.data_transformation_config = DataTransformationConfig()


#         self.utils =  MainUtils()
        
    
    
#     @staticmethod
#     def get_data(feature_store_file_path:str) -> pd.DataFrame:
#         """
#         Method Name :   get_data
#         Description :   This method reads all the validated raw data from the feature_store_file_path and returns a pandas DataFrame containing the merged data. 
        
#         Output      :   a pandas DataFrame containing the merged data 
#         On Failure  :   Write an exception log and then raise an exception
        
#         Version     :   1.2
#         Revisions   :   moved setup to cloud
#         """
#         try:
#             data = pd.read_csv(feature_store_file_path)
#             data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)


#             return data
        
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def get_data_transformer_object(self):
#         try:
            

#             # define the steps for the preprocessor pipeline
#             imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
#             scaler_step = ('scaler', RobustScaler())

#             preprocessor = Pipeline(
#                 steps=[
#                 imputer_step,
#                 scaler_step
#                 ]
#             )
            
#             return preprocessor

#         except Exception as e:
#             raise CustomException(e, sys)
        

             
#     def initiate_data_transformation(self) :
#         """
#             Method Name :   initiate_data_transformation
#             Description :   This method initiates the data transformation component for the pipeline 
            
#             Output      :   data transformation artifact is created and returned 
#             On Failure  :   Write an exception log and then raise an exception
            
#             Version     :   1.2
#             Revisions   :   moved setup to cloud
#         """

#         logging.info(
#             "Entered initiate_data_transformation method of Data_Transformation class"
#         )

#         try:
#             dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
           
            
            
#             X = dataframe.drop(columns= TARGET_COLUMN)
#             y = np.where(dataframe[TARGET_COLUMN]==-1,0, 1)  #replacing the -1 with 0 for model training
            
            
#             X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 )



#             preprocessor = self.get_data_transformer_object()

#             X_train_scaled =  preprocessor.fit_transform(X_train)
#             X_test_scaled  =  preprocessor.transform(X_test)

            


#             preprocessor_path = self.data_transformation_config.transformed_object_file_path
#             os.makedirs(os.path.dirname(preprocessor_path), exist_ok= True)
#             self.utils.save_object( file_path= preprocessor_path,
#                         obj= preprocessor)

#             train_arr = np.c_[X_train_scaled, np.array(y_train) ]
#             test_arr = np.c_[ X_test_scaled, np.array(y_test) ]

#             return (train_arr, test_arr, preprocessor_path)
        

#         except Exception as e:
#             raise CustomException(e, sys) from e







# import sys
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import Pipeline
# from src.constant import *
# from src.exception import CustomException
# from src.logger import logging
# from src.utils.main_utils import MainUtils
# from dataclasses import dataclass

# @dataclass
# class DataTransformationConfig:
#     artifact_dir = os.path.join(artifact_folder)
#     transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
#     transformed_test_file_path = os.path.join(artifact_dir, 'test.npy') 
#     transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')

# class DataTransformation:
#     def __init__(self, feature_store_file_path):
#         self.feature_store_file_path = feature_store_file_path
#         self.data_transformation_config = DataTransformationConfig()
#         self.utils = MainUtils()
    
#     @staticmethod
#     def get_data(feature_store_file_path: str) -> pd.DataFrame:
#         try:
#             data = pd.read_csv(feature_store_file_path, low_memory=False)
#             data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
#             print("Data loaded successfully. Data types:")
#             print(data.dtypes)
#             print("Data sample:")
#             print(data.head())
#             return data
#         except Exception as e:
#             raise CustomException(e, sys)
        
#     def get_data_transformer_object(self):
#         try:
#             imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
#             scaler_step = ('scaler', RobustScaler())
#             preprocessor = Pipeline(steps=[imputer_step, scaler_step])
#             return preprocessor
#         except Exception as e:
#             raise CustomException(e, sys)
    
#     def initiate_data_transformation(self):
#         logging.info("Entered initiate_data_transformation method of Data_Transformation class")
#         try:
#             dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
#             print("Initial data frame:")
#             print(dataframe.head())

#             # Coerce to numeric and handle non-numeric columns
#             dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
#             non_numeric_cols = dataframe.columns[dataframe.isna().any()].tolist()
#             print(f"Non-numeric columns or columns with NaNs: {non_numeric_cols}")

#             # Impute missing values for all columns
#             imputer = SimpleImputer(strategy='constant', fill_value=0)
#             dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
#             print("Data after imputation:")
#             print(dataframe.head())

#             X = dataframe.drop(columns=TARGET_COLUMN)
#             y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)  # replacing -1 with 0 for model training

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#             preprocessor = self.get_data_transformer_object()
#             X_train_scaled = preprocessor.fit_transform(X_train)
#             X_test_scaled = preprocessor.transform(X_test)

#             preprocessor_path = self.data_transformation_config.transformed_object_file_path
#             os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
#             self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)

#             train_arr = np.c_[X_train_scaled, np.array(y_train)]
#             test_arr = np.c_[X_test_scaled, np.array(y_test)]

#             return train_arr, test_arr, preprocessor_path
#         except Exception as e:
#             raise CustomException(e, sys) from e






import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

# Assuming these are defined in the src package
from src.constant import TARGET_COLUMN, artifact_folder
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(artifact_folder)
    transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.npy')
    transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()
    
    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path, low_memory=False)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
            print("Data loaded successfully. Data types:")
            print(data.dtypes)
            print("Data sample:")
            print(data.head())
            return data
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_data_transformer_object(self):
        try:
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())
            preprocessor = Pipeline(steps=[imputer_step, scaler_step])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")
        try:
            # Load and prepare data
            logging.info(f"Loading data from {self.feature_store_file_path}")
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
            logging.info("Initial data frame loaded successfully")
            print("Initial data frame:")
            print(dataframe.head())

            # Convert all columns to numeric, but preserve the original DataFrame shape
            dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
            logging.info("Converted all columns to numeric")
            
            # Separate features and target
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN]
            logging.info("Separated features and target")


            # Handle missing values for X
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            X_imputed = imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns)
            logging.info("Handled missing values in features")

            print("Data after imputation:")
            print(X.head())

            # Handle target variable
            y = np.where(y == -1, 0, 1)  # replacing -1 with 0 for model training
            logging.info("Transformed target variable")


            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            logging.info("Split data into training and testing sets")

            # Get the preprocessor
            preprocessor = self.get_data_transformer_object()
            logging.info("Obtained data transformer object")


            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            logging.info("Scaled training and testing data")

            # Save the preprocessor
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)
            logging.info(f"Saved preprocessor object at {preprocessor_path}")


            # Combine scaled features and target
            train_arr = np.c_[X_train_scaled, y_train]
            test_arr = np.c_[X_test_scaled, y_test]
            logging.info("Combined scaled features and target")

            logging.info("Exited initiate_data_transformation method of Data_Transformation class")
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            logging.error("Error in initiate_data_transformation method of Data_Transformation class")
            raise CustomException(e, sys) from e










