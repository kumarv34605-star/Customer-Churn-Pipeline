import os
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train and Test data read successfully!")
             
            train_df["TotalCharges"] = pd.to_numeric(
                train_df["TotalCharges"],
                errors="coerce"
            )
            
            test_df["TotalCharges"] = pd.to_numeric(
                test_df["TotalCharges"],
                errors="coerce"
            )
            
            ## For training
            X_train = train_df.drop(columns=["Churn", "customerID"])
            y_train = train_df["Churn"]
            
            ## For testing
            X_test = test_df.drop(columns=["Churn", "customerID"])
            y_test = test_df["Churn"]
            
            
            ## Building numerical pipeline
            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            ## Building categorical pipeline
            cat_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown= "ignore"))
                ]
            )
            
            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]
            
            categorical_columns = [
                    "gender",
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod"
                ]
            ## Building preprocessor object ColumnTransformer
            
            preprocessor = ColumnTransformer(
                transformers= [
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )
            
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path, obj= preprocessor
            )
            logging.info("Preprocessor object saved successfully")
            
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            
            logging.info("Data transformation completed successfully")
            
            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test
            )
            
        except Exception as e:
            raise CustomException(e, sys)
            
    if __name__ == "main":
        obj = DataTransformation()