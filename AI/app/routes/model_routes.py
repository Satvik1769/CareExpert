from flask import Blueprint, request, jsonify
import uuid
from ..database.db_connect import connection
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
import firebase_admin
from firebase_admin import  auth

# Database setup
client = connection()
users = client.collection("users")
user_data = client.collection("user_data")

base_dir = os.path.dirname(__file__)
print(list(os.walk(base_dir)))
# Load the pre-trained model, scaler, and label encoder
ml = joblib.load(os.path.join(base_dir, "xgboost_model.joblib"))
scaler = joblib.load(os.path.join(base_dir, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(base_dir, "label_encoder.joblib"))
columns = joblib.load(os.path.join(base_dir, "X_train_columns.joblib"))


# Create a blueprint for user routes
model_bp = Blueprint('auth', __name__)

@model_bp.route('/data', methods=['POST'])
def add_patient():
    data = request.json
    required_fields = ["fever", "cough", "fatigue", "difficulty_breathing", "age", "gender", "blood_pressure", "cholesterol"]

    # Check if all required fields are in the request
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Generate a unique ID for each entry
    patient_id = str(uuid.uuid4())

    # Data to be stored
    patient_data = {
        "fever": data["fever"],
        "cough": data["cough"],
        "fatigue": data["fatigue"],
        "difficulty_breathing": data["difficulty_breathing"],
        "age": int(data["age"]),
        "gender": data["gender"],
        "blood_pressure": data["blood_pressure"],
        "cholesterol": data["cholesterol"],
        "created_at": datetime.now().isoformat()  # Timestamp for record creation
    }

    # Store data in Firebase
    users.document(patient_id).set(patient_data)
    user_input = {
        'Age': patient_data["age"],
        'Fever_Yes': True if patient_data["fever"] == 'Yes' else False,
        'Cough_Yes': True if patient_data["cough"] == 'Yes' else False,
        'Fatigue_Yes': True if patient_data["fatigue"]== 'Yes' else False,
        'Difficulty Breathing_Yes': True if patient_data["difficulty_breathing"]== 'Yes' else False,
        'Gender_Male': 1 if patient_data["gender"] == 'Male' else 0,
    }

    user_input['Blood Pressure_Low'] = 1 if patient_data["blood_pressure"] == 'low' else 0
    user_input['Blood Pressure_Normal'] = 1 if patient_data["blood_pressure"] == 'normal' else 0
    user_input['Blood Pressure_High'] = 1 if patient_data["blood_pressure"] == 'high' else 0
    user_input['Cholesterol Level_Low'] = 1 if patient_data["cholesterol"] == 'low' else 0
    user_input['Cholesterol Level_Normal'] = 1 if patient_data["cholesterol"] == 'normal' else 0
    user_input['Cholesterol Level_High'] = 1 if patient_data["cholesterol"] == 'high' else 0

    # Convert to DataFrame
    user_input_df = pd.DataFrame([user_input])
    print(user_input_df)
    # One-hot encode the categorical features
    user_input_df = pd.get_dummies(user_input_df, drop_first=True)

    # Ensure the columns match with the training data columns
    user_input_df = user_input_df.reindex(columns=columns, fill_value=0)

    # Scale the input
    user_input_scaled = scaler.transform(user_input_df)
    predicted_class = ml.predict(user_input_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    return jsonify({"message": f"You have {predicted_label[0]}"}), 201

@model_bp.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')

    # Check if email already exists in Firestore
    query = user_data.where('email', '==', email).get()
    if query:
        return jsonify({"error": "User already exists"}), 400

    # Hash the password and save to Firestore
    user_data.document(email).set({
        'email': email,
        'password': password,
        'name': name
    })

    return jsonify({"message": "User created successfully", "status": 201}), 201

# Login Route
@model_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Reference the 'user_data' collection in Firestore
    
    # Check if email exists in Firestore
    user_doc = user_data.document(email).get()
    if not user_doc.exists:
        return jsonify({"message": "User not found", "status": 404}), 404

    # Get the stored hashed password from Firestore
    data = user_doc.to_dict()
    stored_password_hash = data.get('password')

    # Verify the password using werkzeug's check_password_hash
    if password == stored_password_hash:
        return jsonify({"message": "Login successful", "status": 200}), 200
    else:
        return jsonify({"message": "Invalid password", "status": 401}), 401
@model_bp.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Welcome to the Model API'}), 200
