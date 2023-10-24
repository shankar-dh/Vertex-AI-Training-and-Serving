from flask import Flask, jsonify, request
from google.cloud import storage
import joblib
import os

app = Flask(__name__)

# Initializing variables
PROJECT_ID = 'weighty-forest-399219' # your project id
model_file_name = "model.pkl"
scaler_file_name = "scaler.pkl"  # Name of your scaler file
bucket_name = 'mlops-data-ie7374' # your bucket name
MODEL_BLOB_NAME = 'model/' + model_file_name
SCALER_BLOB_NAME = 'model/' + scaler_file_name  # Assuming scaler is in the same directory as model on GCS

# Initialize a client
storage_client = storage.Client()
# Create a bucket object for our bucket
bucket = storage_client.get_bucket(bucket_name)

# Create a blob object from the filepath for model
model_blob = bucket.blob(MODEL_BLOB_NAME)
# Download the model file to a destination
model_blob.download_to_filename(model_file_name)

# Create a blob object from the filepath for scaler
scaler_blob = bucket.blob(SCALER_BLOB_NAME)
# Download the scaler file to a destination
scaler_blob.download_to_filename(scaler_file_name)

# Load the model and scaler
model = joblib.load(model_file_name)
scaler = joblib.load(scaler_file_name)  # Load the scaler

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    request_json = request.get_json()
    request_instances = request_json['instances']

    # Assuming each instance has the features as keys
    formatted_instances = [[
        instance['PT08.S1(CO)'],
        instance['NMHC(GT)'],
        instance['C6H6(GT)'],
        instance['PT08.S2(NMHC)'],
        instance['NOx(GT)'],
        instance['PT08.S3(NOx)'],
        instance['NO2(GT)'],
        instance['PT08.S4(NO2)'],
        instance['PT08.S5(O3)'],
        instance['T'],
        instance['RH'],
        instance['AH']
    ] for instance in request_instances]

    # Scale the instances using the loaded scaler
    scaled_instances = scaler.transform(formatted_instances)

    # Make predictions with the model on the scaled instances
    prediction = model.predict(scaled_instances)
    prediction = prediction.tolist()
    output = {'predictions': [{'result': pred} for pred in prediction]}
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
