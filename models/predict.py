from flask import Flask, jsonify, request
from google.cloud import storage
import pickle



app = Flask(__name__)



def load_model_from_gcs(bucket_name, blob_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Download the model to a local file
    blob.download_to_filename('model.pkl')
    
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    return model


@app.route('/predict', methods=['POST'])
def predict():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    data = request.get_json()
    features = data['features']

    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)