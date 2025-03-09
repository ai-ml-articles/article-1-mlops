import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import csv
import io
import logging

class PreprocessOptions(PipelineOptions):
    pass

def preprocess(line):
    """Ensure the preprocessed data is valid before writing."""
    fields = [
        "longitude", "latitude", "housing_median_age", "total_rooms", 
        "total_bedrooms", "population", "households", "median_income", "median_house_value"
    ]

    import csv, io
    reader = csv.reader(io.StringIO(line))
    try:
        row = next(reader)
        if row[0] == "longitude":  # Skip header row
            return None

        row = [float(value) for value in row]  # Convert all values to float
        print(f"Processed row: {row}")  # Debugging step

        return ",".join(map(str, row))
    except Exception as e:
        print(f"Error processing row: {line} -> {e}")
        return None


def run():
    pipeline_options = PipelineOptions(
    runner='DataflowRunner',  # Change from DirectRunner to DataflowRunner
    project='mlops-housing-project',
    temp_location='gs://mlops-housing-bucket/temp/',
    region='us-central1')
    with beam.Pipeline(options=pipeline_options) as p:
        (p
         | "ReadData" >> beam.io.ReadFromText("gs://mlops-housing-bucket/california_housing.csv", skip_header_lines=1)
         | "Preprocess" >> beam.Map(preprocess)
         | "Filter None" >> beam.Filter(lambda x: x is not None)
         | "WriteResults" >> beam.io.WriteToText("gs://mlops-housing-bucket/preprocessed_data/data", file_name_suffix=".csv", shard_name_template=""))

if __name__ == '__main__':
    run()

