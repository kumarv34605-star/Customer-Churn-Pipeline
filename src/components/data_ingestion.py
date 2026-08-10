import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join(
        "notebook",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "raw.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(
                self.ingestion_config.source_data_path
            )

            logging.info("Dataset read successfully")

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logging.info("Train test split completed")

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Train and test data saved successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)