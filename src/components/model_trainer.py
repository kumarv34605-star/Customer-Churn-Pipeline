import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):
        try:
            
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    random_state=42
                )
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_name = max(
                model_report,
                key=lambda model_name: model_report[model_name]["cv_f1"]
            )

            best_model = models[best_model_name]

            best_model_info = model_report[best_model_name]

            logging.info(
                f"Best model found: {best_model_name}"
            )

            logging.info(
                f"Best CV F1: {best_model_info['cv_f1']:.4f}"
            )

            logging.info(
                f"Test F1: {best_model_info['test_f1']:.4f}"
            )

            logging.info(
                f"Best Parameters: {best_model_info['best_params']}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)