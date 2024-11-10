from pymongo import MongoClient
from pymongo.server_api import ServerApi
import firebase_admin
from firebase_admin import credentials, firestore
import os
# MongoDB connection (Replace with your MongoDB URI)

def mongo():

    client = MongoClient(os.getenv("DATABASE"), server_api=ServerApi(
    version="1", strict=True, deprecation_errors=True))
    return client

def connection():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the service account JSON file
    json_path = os.path.join(current_dir, "serviceAccountKey.json")
    

    if not firebase_admin._apps:
        cred = credentials.Certificate(json_path)  # Path to the service account key JSON file
        firebase_admin.initialize_app(cred)

    # Get Firestore client
    db = firestore.client()
    return db



