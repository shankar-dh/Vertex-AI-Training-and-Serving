from google.cloud import storage
from datetime import datetime
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import logging

print("---------------------Lopaliki OCha----------------")

def load_data():

    client = storage.Client()
    bucket_name = 'dataflow-apache-quickstart_weighty-forest-399219' # Change this to your bucket name
    blob_path = 'dhanushkumar13@gmail.com/jobrun/train_data.csv/2023-10-20_18-17-57_00000000' # Change this to your blob path where the data is stored
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Download the content of the file as a string
    data = blob.download_as_text()
    df = pd.read_csv(StringIO(data))
    column_names = [
        'Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]

    df.columns = column_names
    print("Loading Data is successfully done # Change by Dheeraj")

    return df

def data_transform(df):
    
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Splitting the data into training and testing sets (80% training, 20% testing)
    train, test = train_test_split(df, test_size=0.2, shuffle=False)

    # Separating features and target variable
    X_train = train.drop(columns=['CO(GT)'])
    y_train = train['CO(GT)']

    X_test = test.drop(columns=['CO(GT)'])
    y_test = test['CO(GT)']
    
    X_train_scaled = (X_train - X_train.mean()) / X_train.std()
    X_test_scaled = (X_test - X_test.mean()) / X_test.std()
    
    # Standard scaling
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    stats = {
    'mean': X_train.mean().to_dict(),
    'std': X_train.std().to_dict(),
    'mean_test': X_test.mean().to_dict(),
    'std_test': X_test.std().to_dict()
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test,stats


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

df = load_data()
X_train, X_test, y_train, y_test, scaler = data_transform(df)
model = train_model(X_train, y_train)

# Save the model and scaler to local files
local_model_path = "model.pkl"
local_scaler_path = "scaler.json"
joblib.dump(model, local_model_path)

with open(local_scaler_path, 'w') as f:
    json.dump(scaler, f)

# json.dump(scaler, local_scaler_path)

# Specify GCS path
# MODEL_DIR = os.getenv("AIP_MODEL_DIR")
# gcs_model_path = os.path.join(MODEL_DIR, "model.pkl")
# gcs_scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# gcs_model_path = "gs://mlops-data-ie7374/model/model.pkl"
# gcs_scaler_path =  "gs://mlops-data-ie7374/model/scaler.pkl"


version = datetime.now().strftime('%d-%m-%Y-%H%M%S') 
gcs_model_path = f"gs://mlops-data-ie7374/model/model/model_{version}.pkl"
gcs_scaler_path = f"gs://mlops-data-ie7374/model/scaler/scaler_{version}.json"
# Upload model and scaler to GCS

storage_client = storage.Client()
bucket_name, blob_path = gcs_model_path.split("gs://")[1].split("/", 1)
bucket = storage_client.bucket(bucket_name)
blob_model = bucket.blob(blob_path)
blob_model.upload_from_filename(local_model_path)

# Update blob path for scaler and upload
blob_path = gcs_scaler_path.split("gs://")[1].split("/", 1)[1]
blob_scaler = bucket.blob(blob_path)
blob_scaler.upload_from_filename(local_scaler_path)


