from flask import Flask, jsonify, request
from google.cloud import storage
import joblib
import os
import json

app = Flask(__name__)

# Initializing variables
PROJECT_ID = 'weighty-forest-399219' # your project id
bucket_name = 'mlops-data-ie7374' # your bucket name


def fetch_latest_model(bucket_name, prefix="model/model_"):
    """Fetches the latest model file from the specified GCS bucket."""
    # List all blobs in the bucket with the given prefix
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # Extract the timestamps from the blob names and identify the blob with the latest timestamp
    blob_names = [blob.name for blob in blobs]
    if not blob_names:
        raise ValueError("No model files found in the GCS bucket.")

    latest_blob_name = sorted(blob_names, key=lambda x: x.split('_')[-1], reverse=True)[0]

    return latest_blob_name

# Initialize a client
storage_client = storage.Client()
# Create a bucket object for our bucket
bucket = storage_client.get_bucket(bucket_name)

# Load normalization stats
SCALER_BLOB_NAME = 'scaler/normalization_stats.json'
scaler_blob = bucket.blob(SCALER_BLOB_NAME)
stats_str = scaler_blob.download_as_text()
stats = json.loads(stats_str)

# Fetching and loading the latest model
latest_model_blob_name = fetch_latest_model(bucket_name)
local_model_file_name = os.path.basename(latest_model_blob_name)
model_blob = bucket.blob(latest_model_blob_name)
model_blob.download_to_filename(local_model_file_name)
model = joblib.load(local_model_file_name)

def normalize_data(instance, stats):
    normalized_instance = {}
    for feature, value in instance.items():
        mean = stats["mean"].get(feature, 0)
        std = stats["std"].get(feature, 1)
        normalized_instance[feature] = (value - mean) / std
    return normalized_instance

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    request_json = request.get_json()
    request_instances = request_json['instances']

    # Normalize and format each instance
    formatted_instances = []
    for instance in request_instances:
        normalized_instance = normalize_data(instance, stats)
        formatted_instance = [
            normalized_instance['PT08.S1(CO)'],
            normalized_instance['NMHC(GT)'],
            normalized_instance['C6H6(GT)'],
            normalized_instance['PT08.S2(NMHC)'],
            normalized_instance['NOx(GT)'],
            normalized_instance['PT08.S3(NOx)'],
            normalized_instance['NO2(GT)'],
            normalized_instance['PT08.S4(NO2)'],
            normalized_instance['PT08.S5(O3)'],
            normalized_instance['T'],
            normalized_instance['RH'],
            normalized_instance['AH']
        ]
        formatted_instances.append(formatted_instance)

    # Make predictions with the model
    prediction = model.predict(formatted_instances)
    prediction = prediction.tolist()
    output = {'predictions': [{'result': pred} for pred in prediction]}
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
