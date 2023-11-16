# Timeseries MLOps Project

This project outlines the steps for setting up a Machine Learning Operations (MLOps) pipeline using Google Cloud services for a time series dataset.

## Project Configuration
Create a `.env` file in the root of your project directory with the following content. This file should not be committed to your version control system so add it to your `.gitignore` file. This file will be used to store the environment variables used in the project. You can change the values of the variables as per your requirements.
```
# Google Cloud Storage bucket name
BUCKET_NAME= [YOUR_BUCKET]

# Google Cloud AI Platform model directory
AIP_MODEL_DIR=gs://[YOUR_BUCKET]/model

# Google Cloud region
REGION=us-east1

# Google Cloud project ID
PROJECT_ID=[YOUR_PROJECT_ID]

# Container URI for training
CONTAINER_URI=us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/timeseries/train:v1

# Container URI for model serving
MODEL_SERVING_CONTAINER_IMAGE_URI=us-east1-docker.pkg.dev/YOUR_PROJECT_ID/timeseries/serve:v1

# Health check route for the AI Platform model
AIP_HEALTH_ROUTE=/ping

# Prediction route for the AI Platform model
AIP_PREDICT_ROUTE=/predict
```
You will get to know about the usage of these variables in the later sections of the project. Replace the placeholders such as `[YOUR_BUCKET]`, `[YOUR_PROJECT_ID]` with the appropriate values relevant to your setup. [YOUR_BUCKET] should be the name of your GCS bucket. [YOUR_PROJECT_ID] should be the name of your GCP project ID.


Ensure you have all the necessary Python packages installed to run the project by using the `requirements.txt` file. Run the following command to install the packages.
```bash
pip install -r requirements.txt
```

## Data Source
The dataset used in this project is acquired from the UCI Machine Learning Repository. You can find the dataset [here](https://archive.ics.uci.edu/dataset/360/air+quality). This data is version tracked by dvc. Refer to the dvc lab on how to use dvc to track the data.

**Note:**
> We are only tracking the raw data using dvc. The preprocessed data is not tracked by dvc. <br>

## Data Preprocessing
The script `data_preprocess.py` automates the preparation of datasets for a machine learning workflow targeting **CO(GT)**. On initial run, data from the first two months is assigned as training data, with the third month for testing. With each subsequent execution, the script incorporates the previous testing dataset into the training data, and the following month's data becomes the new test set. This ensures an evolving training dataset that benefits from the most recent historical data and a test dataset that is always up-to-date.

**Training and Testing Dataset Updates**  
- **First run:** Training = Months 1-2, Testing = Month 3.
- **Subsequent runs:** Training = Previous Training + Previous Testing, Testing = Next Month.

**Normalization**  
Normalization statistics are computed from the training data and stored in GCS at `gs://YOUR_BUCKET/scaler/normalization_stats.json`. These statistics are updated in tandem with the datasets to reflect the current scale of the training data, providing consistency in the data fed into the model.


- To push the file into the GCS ensure that you have created a service account with Storage Admin role and downloaded the json file. Then run the following command to authenticate the service account.
```bash
gcloud auth activate-service-account --key-file=service_account.json
```
**Note:** 
>You would have already done this step when you would have configured the dvc pipeline. If you have not done that please refer to the dvc lab. If you havent done this step you would get errors while saving the file in GCS <br>

## Model Training, Serving and Building
1. **Folder Structure**:
    - The main directory is `src`.
    - Inside `src`, there are two main folders: `trainer` and `serve`.
    - Each of these folders has their respective Dockerfiles, and they contain the `train.py` and `predict.py` code respectively.
    - There's also a `build.py` script in the main directory which uses the `aiplatform` library to build and deploy the model to the Vertex AI Platform.
```
src/
|-- trainer/
|   |-- Dockerfile
|   |-- train.py
|-- serve/
|   |-- Dockerfile
|   |-- predict.py
|-- build.py
```

### Model Training Pipeline (`trainer/train.py`)
This script orchestrates the model training pipeline, which includes data retrieval, preprocessing, model training, and exporting the trained model to Google Cloud Storage (GCS).

#### Detailed Workflow
- **Data Retrieval**: It starts by fetching the dataset from a GCS bucket using the `load_data()` function. The dataset is then properly formatted with the correct column headers.
- **Preprocessing**: The `data_transform()` function is responsible for converting date and time into a datetime index, splitting the dataset into training and validation sets, and normalizing the data using the pre-computed statistics stored in GCS.
- **Model Training**: A RandomForestRegressor model is trained on the preprocessed data with the `train_model()` function, which is designed to work with the features and target values separated into `X_train` and `y_train`.
- **Model Exporting**: Upon successful training, the model is serialized to a `.pkl` file using `joblib` and uploaded back to GCS. The script ensures that each model version is uniquely identified by appending a timestamp to the model's filename.
- **Normalization Statistics**: The normalization parameters used during preprocessing are retrieved from a JSON file in GCS, ensuring that the model applies consistent scaling to any input data.
- **Version Control**: The script uses the current time in the US/Eastern timezone to create a version identifier for the model, ensuring traceability and organization of model versions.

#### Notes on Environment Configuration:
- A `.env` file is expected to contain environment variables `BUCKET_NAME` and `AIP_MODEL_DIR`, which are crucial for pointing to the correct GCS paths.
- The Dockerfile within the `trainer` directory specifies the necessary Python environment for training, including all the dependencies to be installed.

#### Docker Environment Configuration
The Dockerfile located within the `trainer` directory defines the containerized environment where the training script is executed. Here's a breakdown of its content:
- **Base Image**: Built from `python:3.9-slim`, providing a minimal Python 3.9 environment.
- **Working Directory**: Set to the root (`/`) for simplicity and easy navigation within the container.
- **Environment Variable**: `AIP_STORAGE_URI` is set with the GCS path where the model will be stored.
- **Copying Training Script**: The training script located in the `trainer` directory is copied into the container.
- **Installing Dependencies**: Essential Python libraries such as `pandas`, `scikit-learn`, and `google-cloud-storage` are installed without caching to keep the image size small.
- **Entry Point**: The container's entry point is set to run the training script, making the container executable as a stand-alone training job.

The provided Docker image is purpose-built for training machine learning models on Google Cloud's [Vertex AI Platform](https://cloud.google.com/vertex-ai?hl=en). Vertex AI is an end-to-end platform that facilitates the development and deployment of ML models. It streamlines the ML workflow, from data analysis to model training and deployment, leveraging Google Cloud's scalable infrastructure.

Upon pushing this Docker image to the Google Container Registry (GCR), it becomes accessible for Vertex AI to execute the model training at scale. The image contains the necessary environment setup, including all the dependencies and the `trainer/train.py` script, ensuring that the model training is consistent and reproducible. 

### Model Serving Pipeline (`serve/predict.py`)
This script handles the model serving pipeline, which includes loading the most recent model from Google Cloud Storage (GCS), preprocessing incoming prediction requests, and providing prediction results via a Flask API.

#### Detailed Workflow
- **Model Loading**: At startup, the `fetch_latest_model` function retrieves the most current model file from the specified GCS bucket, determined by the latest timestamp in the model file names.
- **Normalization Statistics**: It loads normalization statistics from a JSON file in GCS, which is used to normalize prediction input data consistently.
- **Flask API**: The Flask app exposes two endpoints: one for health checks (`/ping`) and another for processing predictions (`/predict`). The health check endpoint ensures that the service is running correctly, while the predict endpoint expects a POST request with input instances.
- **Data Preprocessing**: Incoming data in the prediction requests are normalized using the `normalize_data` function, which applies the previously loaded normalization statistics.
- **Prediction**: The preprocessed data is then fed into the RandomForestRegressor model to predict the output, which is returned as a JSON response.

#### Environment Configuration Notes:
- The environment variables `AIP_STORAGE_URI`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE`, and `AIP_HTTP_PORT` are set within the Dockerfile and are crucial for defining the GCS path, API routes, and the port the Flask app runs on.
- The Docker environment configuration specifies the necessary Python environment for the Flask application, including all dependencies that are installed.

#### Docker Environment Configuration
- **Base Image**: Utilizes `python:3.9-slim` as a minimal Python 3.9 base image.
- **Working Directory**: The working directory in the container is set to `/app` for a clean workspace.
- **Environment Variables**: Sets up environment variables needed for the Flask app to correctly interface with Google Cloud services and define API behavior.
- **Copying Serving Script**: The `predict.py` script is copied from the local `serving` directory into the container's `/app` directory.
- **Installing Dependencies**: Installs Flask, `google-cloud-storage`, `joblib`, `scikit-learn`, and `grpcio` using `pip` with no cache to minimize the Docker image size.
- **Entry Point**: The Dockerfile's entry point is configured to run the `predict.py` script, which starts the Flask application when the container is run.

### Pushing Docker Images to Google Artifact Registry
Download Google cloud SDK based on your OS from [here](https://cloud.google.com/sdk/docs/install) and ensure Docker daemon is running. Follow the below steps to push the docker images to Google Artifact Registry.

- **Step 1: Enable APIs**
    1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
    2. Navigate to **APIs & Services** > **Library**.
    3. Search for and enable the **Container Registry API** and the **Vertex AI API**.

- **Step 2: Set Up Google Cloud CLI**
    1. Install and initialize the Google Cloud CLI to authenticate and interact with Google Cloud services from the command line.
        ```bash
        gcloud auth login
        gcloud config set project [YOUR_PROJECT_ID]
        ```

- **Step 3: Configure Docker for GCR**
    1. Give the service account (the one you used for dvc which has bucket access) Artifact Registry Administrator permissions to the project:
    Go to IAM & Admin > IAM and add the service account. Find your service account in the list of service accounts and click the pencil icon to edit. Add the Artifact Registry Administrator role to the service account. Download the JSON key for the service account and save it as `key.json` in the root directory of the project.
        ```bash
        gcloud auth activate-service-account --key-file=key.json
        ```
    2. Configure Docker to use the gcloud command-line tool as a credential helper:
        ```bash
        gcloud auth configure-docker us-east1-docker.pkg.dev
        ```

- **Step 4: Create a GCR Account and Repository Folders**
    1. In the Google Cloud Console, open the Artifact Registry.
    2. Create repositories using a naming convention for `[FOLDER_NAME]`.

- **Step 5: Build and Push the Training Image**
    1. Navigate to the src directory and run:

    ```bash
    docker build -f trainer/Dockerfile -t us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/train:v1 trainer/
    docker push us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/train:v1
    ```

- **Step 6: Build and Push the Serving Image**
    1. Navigate to the src directory and run:

    ```bash
    docker build -f serve/Dockerfile -t us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/serve:v1 serve/
    docker push us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/serve:v1
    ```

    **Note:** 
    > Run the above docker bash codes inside the src directory. <br>

**Reference:** <br>
[Create a custom container image for training (Vertex AI)](https://cloud.google.com/vertex-ai/docs/training/create-custom-container)

### Building and Deploying the Model (`build.py`)

The `build.py` script is responsible for [building and deploying the model to the Vertex AI Platform](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomContainerTrainingJob
). It uses the `aiplatform` library to create a custom container training job and deploy the model to an endpoint.  The CustomContainerTrainingJob class is a part of Google Cloud's Vertex AI Python client library, which allows users to create and manage custom container training jobs for machine learning models. A custom container training job enables you to run your training application in a Docker container that you can customize.

#### Steps to Build and Deploy the Model
1. **Initialize Environment Variables**: The script starts by loading the environment variables such as the region, project ID, bucket, and container URIs using the `initialize_variables` function.

2. **Initialize AI Platform**: With the `initialize_aiplatform` function, the Google Cloud AI platform is initialized using the project ID, region, and staging bucket details.

3. **Create and Run Training Job**: The `create_and_run_job` function creates a `CustomContainerTrainingJob` using the specified container URIs and then starts the training job.

4. **Deploy Model**: After the training job completes, the `deploy_model` function deploys the trained model to an AI Platform endpoint, using the provided model display name.

- When you run this code the aiplatform library will create a custom container training job and deploy the model to an endpoint. It uses both the dockerfiles to build the training and serving images. The training image is used to train the model and the serving image is used to serve the model.
- Once the model is deployed to the endpoint you can use the prediction code to get the predictions from the model. This is online prediction so you can give API requests to the model and get the predictions. The prediction route expects a POST request with the input instances. The input instances to the endpoint should be in the following format:
```
{
    "instances": [
        {
            "Time": "18.00.00",
            "CO(GT)": 2.6,
            "PT08.S1(CO)": 1360,
            "NMHC(GT)": 150,
            "C6H6(GT)": 11.9,
            "PT08.S2(NMHC)": 1046,
            "NOx(GT)": 166,
            "PT08.S3(NOx)": 1056,
            "NO2(GT)": 113,
            "PT08.S4(NO2)": 1692,
            "PT08.S5(O3)": 1268,
            "T": 13.6,
            "RH": 48.9,
            "AH": 0.7578
        }
    ]
}
```

**Note**:
> There are multiple configuration options available in vertex AI other than these parameters. Please refer to the [documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomContainerTrainingJob) for more information. <br>

The advantages of using a custom training job over a normal training job in Vertex AI include:
- Greater flexibility with container images, allowing for specific dependencies and configurations.
- More control over the training environment and resources.
- Ability to integrate with existing CI/CD workflows and toolchains.
- Custom training logic and advanced machine learning experimentation that might not be supported by standard training options.


## Continous Model retraining with Airflow
### Workflow Overview:<br><br>

1. **DAG Configuration**: <br>
    - **DAG Name**: `Retraining_Model`<br>
    - **Description**: Data preprocessing and model retraining at 9 PM every day.<br>
    - **Schedule**: Every day at 9 PM (`schedule_interval='0 21 * * *'`)<br>
    - **Start Date**: October 24, 2023<br>
    - **Retries**: If the task fails, it will retry once (`retries=1`) with a delay of 5 minutes between retries (`retry_delay=dt.timedelta(minutes=5)`).<br><br>

2. **Tasks in the Workflow**:<br>

    a. **Pull preprocess.py from GitHub**:<br>
        - **Task ID**: `pull_preprocess_script`<br>
        - **Action**: Uses the `curl` command to download the `preprocess.py` script from a GitHub repository. This script contains the data preprocessing logic.<br>
        - **GitHub URL**: URL path for preprocess.py code in your Github Repository.<br>
        - **Local Path**: The script is saved to `/tmp/preprocess.py` on the local system.<br><br>

    b. **Execute the Preprocessing Script**:<br>
        - **Task ID**: `run_preprocess_script`<br>
        - **Action**: Executes the previously downloaded Python script (`preprocess.py`) to preprocess the data.<br>
        - **Environment Variable**:<br>
            - `'AIP_MODEL_DIR': 'gs://mlops-data-ie7374/model/'`: Indicates the directory in Google Cloud Storage where the model will be saved after training. (Note: Adjust the path if necessary)<br>
        - **Execution Command**: `python /tmp/preprocess.py`<br><br>

    c. **Pull train.py from GitHub**:<br>
        - **Task ID**: `pull_train_script`<br>
        - **Action**: Uses the `curl` command to download the `train.py` script from a GitHub repository after the preprocessing is done. This is the script that contains the model training logic.<br>
        - **GitHub URL**: URL path for train.py code in your Github Repository.<br>
        - **Local Path**: The script is saved to `/tmp/train.py` on the local system.<br><br>

    d. **Execute the Training Script**:<br>
        - **Task ID**: `run_train_script`<br>
        - **Action**: Executes the previously downloaded Python script (`train.py`) to retrain the model.<br>
        - **Execution Command**: `python /tmp/train.py`<br><br>

3. **Task Dependencies**:<br>
    - First, the `pull_preprocess_script` task is executed to fetch the preprocessing script.<br>
    - Upon its successful completion, the `run_preprocess_script` task is triggered to preprocess the data.<br>
    - In parallel, the `pull_train_script` task fetches the training script, but it will not run until the preprocessing is completed.<br>
    - Once preprocessing is done, the `run_train_script` task is executed to retrain the model. This ensures that the model is trained with the latest preprocessed data.<br><br>

The updated script ensures a streamlined process where data is preprocessed and the model is retrained daily at 9 PM. The retrained model is then saved to Google Cloud Storage, ready for deployment or further evaluation.<br>



To use the latest model for serving rebuild the training image and use the same image for serving. Our prediction code will automatically use the latest model for serving.
Run 'python build.py' to build and deploy the latest model in the Vertex AI Platform.

![Prediction_Page](image.png)

If all the steps are followed correctly, you should be able to see the above page. You can enter the values for the features and click on the predict button to get the prediction for the CO(GT) feature.


