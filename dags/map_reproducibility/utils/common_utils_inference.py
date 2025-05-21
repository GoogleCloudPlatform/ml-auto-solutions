# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jsonlines, json
from google.cloud.bigquery import Client, LoadJobConfig, SourceFormat, SchemaField

PROJECT = "supercomputer-testing"
BUCKET_NAME = "regression-testing-xlml"


def helm_apply_cmds_internal_run_inference(
    framework: str,
    hypercomputer: str,
    config_file,
    recipe_repo_root,
    values_file_path,
    docker_image,
    aotc: bool = False,
    cluster_name: str = "a3plus-benchmark",
    kueue_name: str = "a3-ultra",
    additional_cmds: str = "",
    test_run=False,
):
  gcs_cmd = ""
  if framework == "maxtext":
    gcs_cmd += f" --set volumes.gcsMounts[0].bucketName={BUCKET_NAME} "

  if hypercomputer == "a3ultra":
    if framework != "maxtext":
      gcs_cmd += f" --set queue={kueue_name} "
  else:
    gcs_cmd += f" --set workload.gcsBucketForDataCataPath={BUCKET_NAME} "

  cluster_cmd = ""
  if framework == "nemo" and hypercomputer == "a3ultra":
    cluster_cmd = f" --set clusterName={cluster_name} "

  run_name_cmd = ""
  if framework == "maxtext":
    run_name_cmd = " --set workload.run_name=$JOB_NAME "

  set_aotc = ""
  if aotc:
    set_aotc = " --set-string workload.aotc=true "

  if test_run:
    helm_template_path = f"/home/airflow/gcs/dags/dags/map_reproducibility/helm-charts/{hypercomputer}/{framework}-inference"
  else:
    helm_template_path = f"{recipe_repo_root}/src/helm-charts/{hypercomputer}/{framework}-inference"

  print(f"helm_template_path is {helm_template_path}")

  helm_cmds = (
      f" helm install -f {values_file_path} "
      "--namespace default "
      "--set namespace=default"
      f" --set-file {framework}_config"
      f"={config_file}"
      " --set workload.image"
      f"={docker_image} "
      f"{cluster_cmd} {run_name_cmd} {gcs_cmd} {set_aotc}"
      f"{additional_cmds}"
      f" $JOB_NAME {helm_template_path}",
  )
  print("*******helm cmd is*******")
  print(helm_cmds)
  return helm_cmds


def get_gcs_output_location(bucket_name=BUCKET_NAME):
  return f"gs://{bucket_name}/maxtext-inference/output"


def copy_inference_output_cmds(
    tmpdir, bucket_name=BUCKET_NAME, project=PROJECT
):
  gcs_location = f"gs://{bucket_name}/maxtext-inference/"

  cmds = (
      f"OUTPUT_FILE={tmpdir}/output.txt",
      f"export LOG_FILE=$(gcloud storage ls {gcs_location} --project={project}  | grep microbenchmark | sort -k 2 -r | head -n 1)",
      'echo "LOG_FILE ${LOG_FILE}"',
      "gcloud storage cp $LOG_FILE $OUTPUT_FILE",
  )
  return cmds


def extract_autoregressive_write_to_jsonl(job_name, input_file, output_file):
  """
    Reads a JSON file, extracts the 'autoregressive' data, and writes them to a JSONL file.

  Args:
      input_file (str): Path to the input text file.
      output_file (str): Path to the output JSONL file.
  """
  try:
    with open(input_file, "r") as f:
      data = json.load(f)
    autoregressive_data = data["autoregressive"]
    autoregressive_data["job_name"] = job_name
    with jsonlines.open(output_file, "w") as writter:
      writter.write(autoregressive_data)
      print(f"Extracted results written to {output_file}")
  except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
  except Exception as e:
    print(f"An error occurred during extract to jsonl: {e}")


def write_jsonl_to_bigquery(
    jsonl_file_path: str,
    project_id: str = PROJECT,
    dataset_id: str = "maxtext_inference",
    table_id: str = "microbenchmark_llama2_70b",
):
  try:
    client = Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    schema = [
        SchemaField("job_name", "STRING", mode="REQUIRED"),
        SchemaField("step_in_ms", "FLOAT", mode="NULLABLE"),
        SchemaField("step_in_ms_per_seq", "FLOAT", mode="NULLABLE"),
        SchemaField("global_batch_size", "FLOAT", mode="NULLABLE"),
        SchemaField(
            "total_throughput_tokens_per_second", "FLOAT", mode="NULLABLE"
        ),
        SchemaField("bw_per_device_GB_per_second", "FLOAT", mode="NULLABLE"),
    ]
    job_config = LoadJobConfig(
        source_format=SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=schema,
    )
    print(f"Uploading {jsonl_file_path} to {table_ref.path}.")
    with open(jsonl_file_path, "rb") as source_file:
      load_job = client.load_table_from_file(
          source_file, table_ref, job_config=job_config
      )
    load_job.result()
    print(
        f"Successfully loaded data from '{jsonl_file_path}' to '{table_ref.path}'."
    )
  except Exception as e:
    print(f"An unexpected error occurred during uploading to bq: {e}")
    return False
