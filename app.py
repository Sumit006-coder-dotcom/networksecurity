import sys
import os
from urllib import response
import pickle
import traceback
from Networksecurity.utils.ml_utils.model.estimator import NetworkModel

import certifi
from httpcore import Response
cs = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo
from Networksecurity.exception.exception import NetworkSecurityException 
from Networksecurity.logging.logger import logging
from Networksecurity.pipeline.training_pipeline import TrainingPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle


from fastapi import FastAPI, File, UploadFile,Request,Response
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
import pandas as pd

from Networksecurity.utils.main_utils.utils import load_object,save_object

client = pymongo.MongoClient(mongo_db_url,tlsCAFile=cs)
from Networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        print(" Training started...")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        print(" Training completed successfully.")
        return Response("Training is successful")
    except Exception as e:
        print(" Exception during training:")
        traceback.print_exc() 
        return Response(f"Training failed due to: {str(e)}", status_code=500)
'''    
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        #print(df)
        preprocessor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model= NetworkModel(preprocessor=preprocessor, model=final_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
        print(df.iloc[0])
        y_pred=network_model.predict(df)
        print(y_pred)
        df['predicted_column']=y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1,0)
        #return df.to_json()
        df.to_csv("prediction_output/predicted.csv")
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("predict.html", {"request": request, "table_html": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)
if __name__ == "__main__":
    app.run(app, host="localhost", port=8000)
'''
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        
        
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        
        # Save the full pipeline object
        save_object("final_model/network_model.pkl", obj=network_model)
        
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        
        df.to_csv("prediction_output/predicted.csv", index=False)
        table_html = df.to_html(classes='table table-striped')
        
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@app.get("/predict")
def predict_route(data: dict):
    try:
        # Convert input to suitable format
        X = [data[key] for key in data.keys()]
        prediction = NetworkModel.predict([X])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise NetworkSecurityException(e, sys)