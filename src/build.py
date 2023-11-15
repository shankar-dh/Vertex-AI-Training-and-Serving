from google.cloud import aiplatform
from dotenv import load_dotenv
import os

load_dotenv()

def initialize_variables():
    """
    Initialize environment variables.
    Returns:
        dict: A dictionary containing the region, project id, bucket, container uri, model serving container image uri, and display name.
    """
    variables = {
        "region": os.getenv("REGION"),
        "project_id": os.getenv("PROJECT_ID"),
        "bucket": os.getenv("AIP_MODEL_DIR"), # Should be same as AIP_STORAGE_URI specified in docker file 
        "container_uri": os.getenv("CONTAINER_URI"), 
        "model_serving_container_image_uri": os.getenv("MODEL_SERVING_CONTAINER_IMAGE_URI"),
        "display_name": 'mlops-timeseries'
    }
    return variables

def initialize_aiplatform(variables):
    """
    Initialize AI Platform.
    Args:
        variables (dict): A dictionary containing the region, project id, and bucket.
    """
    aiplatform.init(project=variables["project_id"], location=variables["region"], staging_bucket=variables["bucket"])

def create_and_run_job(variables):
    """
    Create a CustomContainerTrainingJob and start run with service account.
    Args:
        variables (dict): A dictionary containing the display name, container uri, and model serving container image uri.
    Returns:
        Model: The model.
    """
    job = aiplatform.CustomContainerTrainingJob(
        display_name=variables["display_name"],
        container_uri=variables["container_uri"],
        model_serving_container_image_uri=variables["model_serving_container_image_uri"],
    )
    model = job.run(model_display_name=variables["display_name"], service_account="")
    return model

def deploy_model(model, display_name):
    """
    Deploy the model to endpoint.
    Args:
        model (Model): The model.
        display_name (str): The display name.
    Returns:
        Endpoint: The endpoint.
    """
    endpoint = model.deploy(
        deployed_model_display_name=display_name, sync=True
    )
    return endpoint

variables = initialize_variables()
initialize_aiplatform(variables)
model = create_and_run_job(variables)
endpoint = deploy_model(model, variables["display_name"])


