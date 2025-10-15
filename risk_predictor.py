from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# --- UPDATED Model Training with 4 features ---
# Sample Data: [attendance_%, last_exam_score, fee_paid (1/0), average_quiz_score]
# Target: [at_risk (1=yes, 0=no)]
X_sample = np.array([
    [95, 88, 1, 92], [85, 75, 1, 85], [92, 81, 1, 88], [78, 65, 1, 75], [65, 55, 1, 60],
    [71, 62, 0, 55], [55, 45, 1, 40], [40, 35, 0, 30], [80, 50, 1, 50], [60, 48, 0, 45]
])
y_sample = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Standardize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_scaled, y_sample)

print("✅ ML Model trained with new quiz data and ready.")


# --- UPDATED API Endpoint for Predictions ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract all four features from the incoming request
        features = np.array([[
            data['attendancePercentage'],
            data['lastExamScore'],
            data['feePaid'],
            data['averageQuizScore'] # <-- Use the new feature
        ]])

        # Scale the new data
        features_scaled = scaler.transform(features)

        # Predict the probability of being "at-risk"
        risk_probability = model.predict_proba(features_scaled)[0][1]

        return jsonify({'risk_probability': round(risk_probability, 4)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)