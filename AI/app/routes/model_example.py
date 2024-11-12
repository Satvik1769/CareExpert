import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score,classification_report
train=pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Disease classification\Disease_symptom_and_patient_profile_dataset.csv")

X_train = train.drop(columns=['Disease'])
Y_train = train['Disease']

# Replace rare classes with 'Other'
# Replace rare classes with 'Other'
class_counts = Y_train.value_counts()
rare_classes = class_counts[class_counts < 5].index
Y_train = Y_train.replace(rare_classes, 'Other')

# List of categorical and numerical features
nc = X_train.select_dtypes(include=np.number).columns  # numerical columns
cc = X_train.select_dtypes(exclude=np.number).columns  # categorical columns

# Encoding categorical features to numerical using one-hot encoding
X_train = pd.get_dummies(X_train, columns=cc, drop_first=True)

# Apply SMOTE only to the features (X_train)
smote = SMOTE(k_neighbors=3)
X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

# Encode the target labels (Y_resampled) to numeric labels
label_encoder = LabelEncoder()
Y_resampled_encoded = label_encoder.fit_transform(Y_resampled)

# Check unique values of the encoded target labels (to confirm it's numeric)
print("Encoded target labels (numeric):", np.unique(Y_resampled_encoded))

# Split into training and validation sets
X_tr, X_val, Y_tr, Y_val = train_test_split(X_resampled, Y_resampled_encoded, test_size=0.2, random_state=42)

# Scaling the training and validation data (use the scaler fitted on X_tr for X_val)
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# Instantiate XGBoost model
ml = XGBClassifier(objective="multi:softmax", n_estimators=100, learning_rate=0.05, device='cuda:0', eval_metric='merror')

# Train the model
ml.fit(X_tr_scaled, Y_tr)

# Make predictions on validation set
pred_val = ml.predict(X_val_scaled)

# Evaluate the model
acc = accuracy_score(Y_val, pred_val)
cr = classification_report(Y_val, pred_val)
print(f"Accuracy: {acc}")
print(f"Classification Report: \n{cr}")

# Cross-validation on the resampled data
cv = cross_val_score(ml, X_resampled, Y_resampled_encoded, cv=50, scoring='accuracy')
print(f"Cross-validation mean accuracy: {cv.mean()}")


def get_user_input():
    """Collect user input for the features"""
    print("Please enter the following information:")

    # Collecting user inputs for each feature (adjust based on your actual features)
    age = int(input("Enter age: "))
    
    # Now asking for Yes/No instead of True/False
    fever = input("Do you have fever? (Yes/No): ").lower() == 'yes'
    cough = input("Do you have cough? (Yes/No): ").lower() == 'yes'
    fatigue = input("Do you have fatigue? (Yes/No): ").lower() == 'yes'
    difficulty_breathing = input("Do you have difficulty breathing? (Yes/No): ").lower() == 'yes'
    
    gender = input("Enter gender (Male/Female): ")
    
    # Blood Pressure: Ask for low, normal, or high
    blood_pressure = input("Enter blood pressure (Low/Normal/High): ").lower()
    while blood_pressure not in ['low', 'normal', 'high']:
        print("Invalid input. Please enter 'Low', 'Normal', or 'High' for blood pressure.")
        blood_pressure = input("Enter blood pressure (Low/Normal/High): ").lower()

    # Cholesterol Level: Ask for low, normal, or high
    cholesterol_level = input("Enter cholesterol level (Low/Normal/High): ").lower()
    while cholesterol_level not in ['low', 'normal', 'high']:
        print("Invalid input. Please enter 'Low', 'Normal', or 'High' for cholesterol level.")
        cholesterol_level = input("Enter cholesterol level (Low/Normal/High): ").lower()

    # Pack the inputs into a dictionary (this should match the order of columns after encoding)
    user_input = {
        'Age': age,
        'Fever_Yes': fever,
        'Cough_Yes': cough,
        'Fatigue_Yes': fatigue,
        'Difficulty Breathing_Yes': difficulty_breathing,
        'Gender_Male': 1 if gender == 'Male' else 0,  # Assuming binary encoding for gender
    }

    # Encoding blood pressure and cholesterol as categorical features with one-hot encoding
    user_input['Blood Pressure_Low'] = 1 if blood_pressure == 'low' else 0
    user_input['Blood Pressure_Normal'] = 1 if blood_pressure == 'normal' else 0
    user_input['Blood Pressure_High'] = 1 if blood_pressure == 'high' else 0

    user_input['Cholesterol Level_Low'] = 1 if cholesterol_level == 'low' else 0
    user_input['Cholesterol Level_Normal'] = 1 if cholesterol_level == 'normal' else 0
    user_input['Cholesterol Level_High'] = 1 if cholesterol_level == 'high' else 0

    # Convert to DataFrame
    user_input_df = pd.DataFrame([user_input])

    # One-hot encode the categorical features just like we did for training data
    user_input_df = pd.get_dummies(user_input_df, drop_first=True)

    # Make sure the user input matches the training data structure (in case there are new categories)
    user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)

    # Scale the input (using the same scaler as before)
    user_input_scaled = scaler.transform(user_input_df)

    return user_input_scaled

# Ask the user for input and make a prediction
user_input_scaled = get_user_input()
predicted_class = ml.predict(user_input_scaled)
predicted_label = label_encoder.inverse_transform(predicted_class)

print(f"The predicted disease class is: {predicted_label[0]}")