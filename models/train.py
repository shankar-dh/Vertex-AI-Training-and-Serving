from google.cloud import storage
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


def load_data():

    client = storage.Client()
    bucket_name = 'dataflow-apache-quickstart_weighty-forest-399219'
    blob_path = 'dhanushkumar13@gmail.com/jobrun/train_data.csv/2023-10-20_18-17-57_00000000'
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

    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train, X_test):
    # Building a Random Forest regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model



def save_model_to_gcs(model, bucket_name, model_filename):
    # Serialize the model to a file
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    # Initialize a GCS client
    storage_client = storage.Client()

    # Reference the bucket and the blob (file location)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_filename)

    # Upload the file to GCS
    blob.upload_from_filename(model_filename)
    print(f"Model saved to {bucket_name}/{model_filename}")


