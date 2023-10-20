import apache_beam as beam
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
import os
import pandas as pd
import json

class LoadDataFn(beam.DoFn):
    def process(self, element):
        file_path = element

        df = pd.read_csv(file_path)
        return [df]

class PreprocessDataFn(beam.DoFn):
    def process(self, element):
        df = element
        df_numeric = df.drop(columns=['Date', 'Time'])
        mean_train = df_numeric.mean()
        std_train = df_numeric.std()
        train_normalized = (df_numeric - mean_train) / std_train
        normalization_stats = {
            'mean': mean_train.to_dict(),
            'std': std_train.to_dict()
        }
        yield (train_normalized, normalization_stats)

class WriteToJson(beam.PTransform):
    def __init__(self, output_file):
        self.output_file = output_file

    def expand(self, pcoll):
        return pcoll | "WriteToJson" >> WriteToText(self.output_file, file_name_suffix=".json", shard_name_template='')
    

from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import SetupOptions  # add this import


def run_pipeline():
    # Define GCP Dataflow options
    # pipeline_options = PipelineOptions()
    # pipeline_options.view_as(GoogleCloudOptions).project = 'weighty-forest-399219'
    # pipeline_options.view_as(GoogleCloudOptions).job_name = 'timseseries'
    # pipeline_options.view_as(GoogleCloudOptions).staging_location = 'gs://mlops-data-ie7374/staging'
    # pipeline_options.view_as(GoogleCloudOptions).temp_location = 'gs://mlops-data-ie7374/temp'
    # pipeline_options.view_as(GoogleCloudOptions).region = 'us-east1'  # replace 'us-central1' with your desired region
    # pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'

     # Define pipeline options for local execution
    pipeline_options = PipelineOptions()
    pipeline_options.view_as(StandardOptions).runner = 'DirectRunner'  # Set to DirectRunner for local execution

    setup_options = pipeline_options.view_as(SetupOptions)
    setup_options.setup_file = os.path.abspath('setup.py')

    with beam.Pipeline(options=pipeline_options) as p:
        # Paths to GCS
        input_data = (p | "ReadInputData" >> beam.Create([
            # 'gs://mlops-data-ie737/data/train/train_data.csv'
            os.path.join("..", "data", "train", "train_data.csv")
        ]))


        loaded_data = (input_data | "LoadData" >> beam.ParDo(LoadDataFn()))

        preprocessed_data = (loaded_data | "PreprocessData" >> beam.ParDo(PreprocessDataFn()))

        # Write the normalized training data to a CSV file.
        (preprocessed_data | "WriteTrainData" >> beam.Map(lambda x: x[0])  
                           | "WriteToCSV" >> beam.Map(lambda df: df.to_csv(
                            #    'gs://mlops-data-ie7374/data/train/train_normalized.csv',
                            os.path.join("..", "data", "train", "train_normalized.csv"),
                                 index=False)))

        # Write normalization stats to a JSON file using Beam's WriteToText transform
        (preprocessed_data | "ExtractNormalizationStats" >> beam.Map(lambda x: json.dumps(x[1]))
                        | "WriteToJson" >> WriteToJson(os.path.join("..", "data", "normalization_stats")))


if __name__ == "__main__":
    run_pipeline()
