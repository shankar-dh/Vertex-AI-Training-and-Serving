from google.cloud import aiplatform

REGION = 'us-east1'
PROJECT_ID = 'weighty-forest-399219'
bucket = 'gs://mlops-data-ie7374/model/' # Should be same as AIP_STORAGE_URI specified in docker file
container_uri='us-east1-docker.pkg.dev/weighty-forest-399219/mlops/train:v1'
model_serving_container_image_uri='us-east1-docker.pkg.dev/weighty-forest-399219/mlops/serve:v1'
display_name='mlops-timeseries'

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket)

# Create CustomContainerTrainingJob and start run with some service account
job = aiplatform.CustomContainerTrainingJob(
display_name=display_name,
container_uri=container_uri,
model_serving_container_image_uri=model_serving_container_image_uri,
)

model = job.run( model_display_name=display_name, service_account="")

# deploy the model to endpoint
endpoint = model.deploy(
deployed_model_display_name=display_name, sync=True
)