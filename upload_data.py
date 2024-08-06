from pymongo.mongo_client import MongoClient
import pandas as pd
import json


uri = "mongodb+srv://The_anjalirai:anjalirai@cluster0.m6qkmnh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri)



# create database name and collection name
DATABASE_NAME="MLProjects"
COLLECTION_NAME = "waferfault"

# read data as a dataframe
df=pd.read_csv(r"C:\MLProjects\Sensor_Product_\notebooks\wafer_23012020_041211.csv")
df=df.drop("Unnamed: 0",axis=1)
# convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

# now dump the data into databse
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)