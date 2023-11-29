FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://mlops___fall2023/model
ENV BUCKET_NAME=mlops___fall2023

COPY trainer /trainer

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio gcsfs python-dotenv


ENTRYPOINT ["python", "-m", "trainer.train"]
