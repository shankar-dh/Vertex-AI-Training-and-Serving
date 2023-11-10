import os
import pandas as pd
import json
from google.cloud import storage

def create_directories():
    # Define local directory paths
    data_dir = os.path.join(os.getcwd(), 'data')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Making sure the directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, test_dir

def preprocess_data(train_data, normalization_stats_json_path):
    train_data_numeric = train_data.drop(columns=['Date', 'Time'])

    # Calculate mean and standard deviation for each feature in the training data
    mean_train = train_data_numeric.mean()
    std_train = train_data_numeric.std()

    # Store normalization statistics in a dictionary
    normalization_stats = {
        'mean': mean_train.to_dict(),
        'std': std_train.to_dict()
    }

    # Save the normalization statistics to a JSON file
    with open(normalization_stats_json_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)

def update_datasets(monthly_dataframes, train_data_csv_path, test_data_csv_path, normalization_stats_json_path):
    # Sorting the months by year and month
    sorted_months = sorted(monthly_dataframes.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))

    try:
        train_data = pd.read_csv(train_data_csv_path)
        test_data = pd.read_csv(test_data_csv_path)

        # Get the last date from the test set to determine which data to add
        last_test_date = pd.to_datetime(test_data['Date']).dt.to_period('M').max()
        last_test_year_month = str(last_test_date)

        # Find the next month in the sorted list of keys from the monthly_dataframes
        next_month_index = sorted_months.index(last_test_year_month) + 1

        # Checking if there is a next month's data to add
        if next_month_index < len(sorted_months):
            # Add the previous test set to the training set
            train_data = pd.concat([train_data, test_data], ignore_index=True)

            # Set the new test set as the next month's data
            test_data = monthly_dataframes[sorted_months[next_month_index]]

        else:
            return train_data, test_data

    except FileNotFoundError:
        # If train/test data does not exist, start from the first month
        if len(sorted_months) >= 3:
            # Start with the first two months for training and the third for testing
            train_data = pd.concat([monthly_dataframes[sorted_months[0]], monthly_dataframes[sorted_months[1]]], ignore_index=True)
            test_data = monthly_dataframes[sorted_months[2]]
        elif len(sorted_months) == 2:
            # If only two months are available, use the first for training and the second for testing
            train_data = monthly_dataframes[sorted_months[0]]
            test_data = monthly_dataframes[sorted_months[1]]
        else:
            # If only one month is available, use it for training and leave test set empty
            train_data = monthly_dataframes[sorted_months[0]]
            test_data = pd.DataFrame()

    # Save the updated datasets
    train_data.to_csv(train_data_csv_path, index=False)
    test_data.to_csv(test_data_csv_path, index=False)

    # Preprocess the training data
    preprocess_data(train_data, normalization_stats_json_path)



def upload_to_gcs(local_path, gcs_path):
    storage_client = storage.Client()
    bucket_name, blob_path = gcs_path.split("gs://")[1].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

def main():
    train_dir, test_dir = create_directories()
    
    # Define paths for the train and test data and normalization stats
    train_data_csv_path = os.path.join(train_dir, 'train_data.csv') # Change this to train csv file in GCS bucket
    test_data_csv_path = os.path.join(test_dir, 'test_data.csv') # Change this to test csv file in GCS bucket
    normalization_stats_json_path = os.path.join(train_dir, 'normalization_stats.json')
    
    air_quality_data = pd.read_excel(r'E:\NEU\TA\Time Series\data\raw_data\AirQualityUCI.xlsx') #Modify this to read from your GCS Bucket
    air_quality_data['YearMonth'] = air_quality_data['Date'].dt.to_period('M')
    monthly_groups = air_quality_data.groupby('YearMonth')
    monthly_dataframes = {str(period): group.drop('YearMonth', axis=1) for period, group in monthly_groups}

    # Update the datasets
    update_datasets(monthly_dataframes, train_data_csv_path, test_data_csv_path, normalization_stats_json_path)

    # Define GCS paths for the data
    gcs_scaler_path = "gs://mlops-data-ie7374/scaler/normalization_stats.json"
    gcs_train_data_path = "gs://mlops-data-ie7374/data/train/train_data.csv"
    gcs_test_data_path = "gs://mlops-data-ie7374/data/test/test_data.csv"

    # Upload the updated files to GCS
    upload_to_gcs(train_data_csv_path, gcs_train_data_path)
    upload_to_gcs(test_data_csv_path, gcs_test_data_path)
    upload_to_gcs(normalization_stats_json_path, gcs_scaler_path)

if __name__ == '__main__':
    main()
