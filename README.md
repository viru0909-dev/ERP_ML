# SmartCampus AI - Risk Analytics Service

This repository contains the source code for the Risk Analytics microservice of the SmartCampus AI project. It is a lightweight Python service that uses a machine learning model to predict the likelihood of a student becoming disengaged or failing.

## ✨ Features

° Predictive REST API: A simple and efficient Flask endpoint that receives student data and returns a predictive risk score.

° Machine Learning Model: Utilizes a pre-trained Logistic Regression model from Scikit-learn to analyze factors like attendance, marks, and fee status.

° Decoupled Microservice: Designed to run independently of the core backend, allowing for separate scaling, updates, and maintenance of the AI engine.

## 🛠️ Tech Stack

° Framework: Flask

° Language: Python

° ML Library: Scikit-learn, Pandas

° Containerization: Docker

## 🚀 Getting Started

### Prerequisites

Python (v3.9 or later)

pip

### Installation
1. Clone the repository:
git clone https://github.com/viru0909-dev/ERP_ML.git
2. Install dependencies:
pip install -r requirements.txt
3. Run the Flask application:
flask run
The service will be available at http://127.0.0.1:5000.

## 🔌 API Usage

Send a POST request to the /predict endpoint with a JSON payload containing student data.

**Endpoint:** /predict
**Method:** POST

**Example Request Body:**

JSON

{
  "attendance": 75.5,
  "marks": 62.0,
  "fees_paid": true
}
**Example Success Response:**

JSON

{
  "risk_probability": 0.68
}
