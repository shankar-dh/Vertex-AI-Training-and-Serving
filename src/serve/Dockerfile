FROM python:3.9-slim

WORKDIR /app

COPY serving/serve.py /app/

RUN pip install Flask google-cloud-storage joblib scikit-learn grpcio

ENV AIP_STORAGE_URI=gs://mlops-data-ie7374/model/
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

ENTRYPOINT ["python", "serve.py"]
