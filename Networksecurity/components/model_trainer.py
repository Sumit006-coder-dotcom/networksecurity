import os
import sys
import pandas as pd
import joblib
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging

from Networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, DataIngestionArtifact
from Networksecurity.entity.config_entity import ModelTrainerConfig

from Networksecurity.utils.ml_utils.model.estimator import NetworkModel
from Networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from Networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow
import mlflow.sklearn


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        try:
            with mlflow.start_run():
                f1_score = classificationmetric.f1_score
                precision_score = classificationmetric.precision_score
                recall_score = classificationmetric.recall_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                mlflow.sklearn.log_model(best_model, "model")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            # Load preprocessor from data transformation artifact
            preprocessor = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            
            preprocessor = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=3)),
    ('scaler', StandardScaler())
])



            # Fit the preprocessor on training data (important to avoid NotFittedError)
            preprocessor.fit(X_train)
            joblib.dump(preprocessor, 'final_model/preprocessor.pkl')

            # Save the fitted preprocessor
            save_object("final_model/preprocessor.pkl",preprocessor)

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "Adaboost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
                "Random Forest": {'n_estimators': [8, 16, 32, 256]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 256]
                },
                "Logistic Regression": {},
                "Adaboost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 256]
                }
            }
            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Evaluate all models and get the best one
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Best model selection
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
        


            # Training metrics
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            # Track training metrics
            self.track_mlflow(best_model, classification_train_metric)

            # Testing metrics
            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, classification_test_metric)

            # Save the trained model and network_model
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)


            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            save_object("final_model/model.pkl", best_model)
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            preprocessor.fit(X_train)

            save_object("final_model/preprocessor.pkl",preprocessor)
            

            # Return ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load training and testing arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)