FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://mlops-data-ie7374/model/

COPY trainer /trainer

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio gcsfs


ENTRYPOINT ["python", "-m", "trainer.train"]
