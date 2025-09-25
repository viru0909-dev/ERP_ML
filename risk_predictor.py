from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# --- Model Training (This happens once on startup) ---
# For the hackathon, we'll train the model with sample data when the server starts.
# In a real-world application, you would load a pre-trained model from a file.

# Sample Data: [attendance_%, last_exam_score, fee_paid (1=yes, 0=no)]
# Target: [at_risk (1=yes, 0=no)]
X_sample = np.array([
    [95, 88, 1], [85, 75, 1], [92, 81, 1], [78, 65, 1], [65, 55, 1],
    [71, 62, 0], [55, 45, 1], [40, 35, 0], [80, 50, 1], [60, 48, 0]
])
y_sample = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = Not At-Risk, 1 = At-Risk

# Standardize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_scaled, y_sample)

print("✅ ML Model trained and ready to make predictions.")


# --- API Endpoint for Predictions ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from the incoming request from your Java backend
        features = np.array([[
            data['attendancePercentage'],
            data['lastExamScore'],
            data['feePaid']
        ]])

        # Scale the new data using the same scaler that the model was trained on
        features_scaled = scaler.transform(features)

        # Predict the probability of being "at-risk" (which is the second value in the array)
        risk_probability = model.predict_proba(features_scaled)[0][1]

        # Return the probability as a JSON response
        return jsonify({'risk_probability': round(risk_probability, 4)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# In risk_predictor.py

if __name__ == '__main__':
    # CHANGE THIS LINE'S HOST
    app.run(host='0.0.0.0', port=5002)