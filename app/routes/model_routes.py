from flask import Blueprint, request, jsonify
import uuid
from ..database.db_connect import connection
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
# Database setup
client = connection()
users = client.collection("users")

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
    required_fields = ["disease", "fever", "cough", "fatigue", "difficulty_breathing", "age", "gender", "blood_pressure", "cholesterol"]

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
        "age": data["age"],
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

    return jsonify({"message": f"The predicted disease class is: {predicted_label[0]}"}), 201

@model_bp.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Welcome to the Model API'}), 200
