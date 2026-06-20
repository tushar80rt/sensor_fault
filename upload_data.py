from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://tushar80rt_db_user:uz9PMN8byvn6yu7J@rag.7uyh4wq.mongodb.net/?appName=RAG"

# Create a new client and connect the server

client = MongoClient(uri)

# create database name and collection name

DATABASE_NAME = "sensor"
COLLECTION_NAME = "waferfault"

df = pd.read_csv("C:\Desktop\sensor fault detection\notebooks\wafer_23012020_041211.csv")
df.head()
df = df.drop("Unnamed: 0" , axis=1)

json_record = list(json.loads(df.T.to_json().values()))

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)