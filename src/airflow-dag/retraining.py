import datetime as dt
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

LOCAL_FILE_PATH = '/tmp/train.py'
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/shankar-dh/Timeseries/main/src/trainer/train.py?token=GHSAT0AAAAAACHMX6YFUS2OOI6WZJNGZQY2ZJ4HONA'  # Adjust the path accordingly

default_args = {
    'owner': 'Time_Series_IE7374',
    'start_date': dt.datetime(2023, 10, 24),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

dag = DAG(
    'Retraining_Model',
    default_args=default_args,
    description='Model retraining at 9 PM everyday',
    schedule_interval='0 21 * * *',  # Every day at 9 pm
    catchup=False,
)

# Task to pull train.py from GitHub
pull_script = BashOperator(
    task_id='pull_script_from_github',
    bash_command=f'curl -o {LOCAL_FILE_PATH} {GITHUB_RAW_URL}',
    dag=dag,
)

env = {
    'AIP_MODEL_DIR': 'gs://mlops-data-ie7374/model/'
}
bash_command = "python /tmp/train.py"

# Operator to execute the Python script
run_script = BashOperator(
    task_id='run_python_script',
    bash_command=bash_command,
    env=env,
    dag=dag,
)


pull_script >> run_script
