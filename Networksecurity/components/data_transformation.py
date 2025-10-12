import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from Networksecurity import pipeline
from Networksecurity.constant.training_pipeline import TARGET_COLUMN
from Networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from Networksecurity.entity.artifact_entity import(
    DataValidationArtifact,
    DataTransformationArtifact
)

from Networksecurity.entity.config_entity import DataTransformationConfig
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def get_data_transformer_object(cls) -> Pipeline:
        """
        It initializes KNNImputer object with the parameters specified in the tarining.py file
        and returns the pipeline object with the KNNImputer object as a step.

        Args:
            cls: DataTransformation

        Returns:
        A Pipeline object with a KNNImputer 
        """

        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialise KNNImputer with: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor:Pipeline= Pipeline([('Imputer',imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            # Get the preprocessor pipeline
            preprocessor_object = self.get_data_transformer_object()

            # Define a consistent path for the preprocessor
            preprocessor_file_path = os.path.join(
                "Artifact",
                "data_transformation",
                "preprocessed_object",
                "preprocessor.pkl"
            )

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(preprocessor_file_path), exist_ok=True)

            # Save the preprocessor object
            save_object(preprocessor_file_path, preprocessor_object)
            logging.info(f"Preprocessor saved at: {preprocessor_file_path}")

            # Read validated train and test data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info("Read train and test data completed")

            # Split input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Fit preprocessor on train and transform both train and test
            preprocessor_object.fit(input_feature_train_df)
            transformed_input_train = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test = preprocessor_object.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[transformed_input_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test, np.array(target_feature_test_df)]

            # Save transformed arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Transformed train and test arrays saved successfully")

            # Create DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                preprocessed_object_file_path=preprocessor_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(f"Data Transformation Artifact created: {data_transformation_artifact}")
            return data_transformation_artifact  # Only one return here

        except Exception as e:
            raise NetworkSecurityException(e, sys)