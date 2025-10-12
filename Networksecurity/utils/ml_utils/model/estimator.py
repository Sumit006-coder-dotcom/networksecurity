from Networksecurity.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
from sklearn.pipeline import Pipeline
import joblib


import os
import sys

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging

import sys
from Networksecurity.exception.exception import NetworkSecurityException

class NetworkModel:
    def __init__(self, preprocessor, model):
        #try:
            #if preprocessor is None or model is None:
                #raise ValueError("Preprocessor and model cannot be None")
            self.preprocessor = preprocessor
            self.model = model
        #except Exception as e:
            #aise NetworkSecurityException(e, sys)
        
    def predict(self, X):
        #try:
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict(X_transformed)
            y_hat = self.model.predict(X_transformed)
            return y_hat
        #except Exception as e:
            #raise NetworkSecurityException(e, sys)