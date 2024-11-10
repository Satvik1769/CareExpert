import joblib
import pandas as pd
import numpy as np
import os


# Base directory where this script is located
base_dir = os.path.dirname(__file__)
print(list(os.walk(base_dir)))
# Load the pre-trained model, scaler, and label encoder
ml = joblib.load(os.path.join(base_dir, "xgboost_model.joblib"))
scaler = joblib.load(os.path.join(base_dir, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(base_dir, "label_encoder.joblib"))
columns = joblib.load(os.path.join(base_dir, "X_train_columns.joblib"))
def get_user_input():
    """Collect user input for the features"""
    print("Please enter the following information:")

    age = int(input("Enter age: "))
    fever = input("Do you have fever? (Yes/No): ").lower() == 'yes'
    cough = input("Do you have cough? (Yes/No): ").lower() == 'yes'
    fatigue = input("Do you have fatigue? (Yes/No): ").lower() == 'yes'
    difficulty_breathing = input("Do you have difficulty breathing? (Yes/No): ").lower() == 'yes'
    gender = input("Enter gender (Male/Female): ")

    blood_pressure = input("Enter blood pressure (Low/Normal/High): ").lower()
    while blood_pressure not in ['low', 'normal', 'high']:
        print("Invalid input. Please enter 'Low', 'Normal', or 'High' for blood pressure.")
        blood_pressure = input("Enter blood pressure (Low/Normal/High): ").lower()

    cholesterol_level = input("Enter cholesterol level (Low/Normal/High): ").lower()
    while cholesterol_level not in ['low', 'normal', 'high']:
        print("Invalid input. Please enter 'Low', 'Normal', or 'High' for cholesterol level.")
        cholesterol_level = input("Enter cholesterol level (Low/Normal/High): ").lower()

    # Pack the inputs into a dictionary
    user_input = {
        'Age': age,
        'Fever_Yes': fever,
        'Cough_Yes': cough,
        'Fatigue_Yes': fatigue,
        'Difficulty Breathing_Yes': difficulty_breathing,
        'Gender_Male': 1 if gender == 'Male' else 0,
    }

    user_input['Blood Pressure_Low'] = 1 if blood_pressure == 'low' else 0
    user_input['Blood Pressure_Normal'] = 1 if blood_pressure == 'normal' else 0
    user_input['Blood Pressure_High'] = 1 if blood_pressure == 'high' else 0
    user_input['Cholesterol Level_Low'] = 1 if cholesterol_level == 'low' else 0
    user_input['Cholesterol Level_Normal'] = 1 if cholesterol_level == 'normal' else 0
    user_input['Cholesterol Level_High'] = 1 if cholesterol_level == 'high' else 0

    # Convert to DataFrame
    user_input_df = pd.DataFrame([user_input])
    print(user_input_df)
    # One-hot encode the categorical features
    user_input_df = pd.get_dummies(user_input_df, drop_first=True)

    # Ensure the columns match with the training data columns
    user_input_df = user_input_df.reindex(columns=columns, fill_value=0)

    # Scale the input
    user_input_scaled = scaler.transform(user_input_df)

    return user_input_scaled

# Ask the user for input and make a prediction
user_input_scaled = get_user_input()
predicted_class = ml.predict(user_input_scaled)
predicted_label = label_encoder.inverse_transform(predicted_class)

print(f"The predicted disease class is: {predicted_label[0]}")
