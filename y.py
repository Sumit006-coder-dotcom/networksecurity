import pandas as pd
import pymongo
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

client = pymongo.MongoClient(MONGO_DB_URL)
db = client["SUMIT"]
collection = db["NetworkData"]

df = pd.read_csv("Network_Data/dataset_full.csv")
data = df.to_dict(orient="records")
collection.insert_many(data)
print(f"✅ Inserted {len(data)} records into MongoDB!")