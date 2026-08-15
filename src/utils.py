import os
import sys
import pickle

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):
    try:
        report = {}

        for i in range(len(list(models))):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            print(f"\n{'=' * 50}")
            print(f"Training: {model_name}")
            print(f"{'=' * 50}")

            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                scoring="f1_weighted",
                verbose=1
            )

            gs.fit(X_train, y_train)

            print(f"Best Parameters: {gs.best_params_}")
            print(f"Best CV F1: {gs.best_score_:.4f}")

            # Set the best hyperparameters found by GridSearchCV
            model.set_params(**gs.best_params_)

            # Train the model with the best parameters
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Train and test performance
            train_model_score = f1_score(
                y_train,
                y_train_pred,
                average="weighted"
            )

            test_model_score = f1_score(
                y_test,
                y_test_pred,
                average="weighted"
            )

            print(f"Train F1: {train_model_score:.4f}")
            print(f"Test F1:  {test_model_score:.4f}")

            # IMPORTANT:
            # Use CV score for model selection.
            # Test score is kept only for final evaluation.
            report[model_name] = {
                "cv_f1": gs.best_score_,
                "train_f1": train_model_score,
                "test_f1": test_model_score,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)