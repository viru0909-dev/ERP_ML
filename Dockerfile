# Dockerfile for Risk_Anaylitics

# Use a slim Python base image for a smaller size
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port your application will run on
EXPOSE 5002

# Run the app using Gunicorn, a production-ready server
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "risk_predictor:app"]