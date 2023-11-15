# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

COPY predict.py /app/

# Install Flask and google-cloud-storage, joblib, scikit-learn, grpcio
RUN pip install Flask google-cloud-storage joblib scikit-learn grpcio

ENV AIP_STORAGE_URI=gs://mlops-data-ie7374/model/
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

# Run serve.py when the container launches
ENTRYPOINT ["python", "serve.py"]
