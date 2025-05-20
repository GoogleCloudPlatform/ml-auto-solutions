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

"""Stand alone file to write XML based test results to a BigQuery Table."""

import sys
import traceback
import datetime
import pprint
import xml.etree.ElementTree as ET
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def check_bigquery_table_exists(project_id, dataset_id, table_id):
  """
  Checks if a BigQuery table exists.

  Args:
      project_id (str): Your Google Cloud project ID.
      dataset_id (str): The ID of the BigQuery dataset.
      table_id (str): The ID of the BigQuery table.

  Returns:
      bool: True if the table exists, False otherwise.
  """
  client = bigquery.Client(project=project_id)
  full_table_id = f"{project_id}.{dataset_id}.{table_id}"

  try:
    client.get_table(full_table_id)  # Make an API request to get table metadata
    print(f"Table '{full_table_id}' exists.")
    return True
  except NotFound:
    print(f"Table '{full_table_id}' does not exist.")
    return False
  except Exception as e:
    print(f"An error occurred: {e}")
    return False


def parse_xml_and_insert_summary_bq(
    tests_exit_code,
    xml_string,
    project_id,
    dataset_id,
    table_id,
    run_type,
    exec_iso8601_datetime,
    compser_env,
    task_id,
    gcs_folder,
    stdout_filename,
    dockerlog_filename,
    html_filename,
) -> bool:
  """
  Parses XML test results and inserts a high level summary of them into a BigQuery table.

  Args:
      xml_string: The XML string containing test results.
      project_id: Your Google Cloud Project ID.
      dataset_id: The BigQuery dataset ID.
      table_id: The BigQuery table ID.
  """
  if xml_string is not None:
    try:
      root = ET.fromstring(xml_string)
    except ET.ParseError as e:
      print(f"Error parsing XML: {e}")
      return False

  client = bigquery.Client(project=project_id)
  dataset_ref = client.dataset(dataset_id)

  # Define the table schema
  schema = [
      bigquery.SchemaField("execution_datetime", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("test_run_datetime", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("test_run_type", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("test_run_status", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("project_id", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("composer_env", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("task_group_id", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("hostname", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("tests", "INTEGER", mode="NULLABLE"),
      bigquery.SchemaField("errors", "INTEGER", mode="NULLABLE"),
      bigquery.SchemaField("failures", "INTEGER", mode="NULLABLE"),
      bigquery.SchemaField("skipped", "INTEGER", mode="NULLABLE"),
      bigquery.SchemaField("urllogfile", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("gcs_folder", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("authed_url_logfile", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("authed_url_html", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("authed_url_xmlresults", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("authed_url_dockerlog", "STRING", mode="NULLABLE"),
  ]

  print(f"{client.list_tables(dataset_ref)}")
  table_ref = bigquery.Table(
      f"{project_id}.{dataset_id}.{table_id}", schema=schema
  )
  if not check_bigquery_table_exists(project_id, dataset_id, table_id):
    # Create table if it doesn't already exist
    try:
      table_ref = client.create_table(table_ref)  # Make an API request.
      print(
          f"Created table {table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"
      )
    except Exception as e:
      print(f"Got Exception {e}")
      print(
          f"The exception class is: {e.__class__.__module__}.{e.__class__.__name__}"
      )
      print(f"{pprint.pprint(e)}")
      return False

  rows_to_insert = []

  # Convert gs:// links to a format such as:
  # https://storage.mtls.cloud.google.com/ml-auto-solutions/output/sparsity_diffusion_devx/jax/bite_tpu_unittest_main-v6e-4-2025-05-08-12-00-08/test_output.txt
  if gcs_folder.startswith("gs://"):
    gcs_path = gcs_folder.removeprefix("gs://")
  else:
    print("Error - GCS folder link does NOT start with gs://")
    return False
  gcs_path = gcs_path.removesuffix("/")
  authed_gcs_logfile_link = f"https://storage.mtls.cloud.google.com/{gcs_path}/axlearn-test-results/{stdout_filename}"
  authed_gcs_html_link = f"https://storage.mtls.cloud.google.com/{gcs_path}/axlearn-test-results/{html_filename}"
  authed_gcs_xmlresults_link = f"https://storage.mtls.cloud.google.com/{gcs_path}/axlearn-test-results/testing.xml"
  authed_gcs_dockerlog_link = f"https://storage.mtls.cloud.google.com/{gcs_path}/axlearn-test-results/{dockerlog_filename}"

  if task_id.find(".") > 0:
    task_group_id = task_id.split(".")[0]
  else:
    task_group_id = task_id

  if xml_string is None:
    # Don't attempt to get details from XML data, just fill in what we can to give
    # the user details to assist with debugging e.g. set pytest timestamp to None
    # as we don't have a reliable value for the pytest start time
    timestamp = None

    rows_to_insert.append({
        "execution_datetime": exec_iso8601_datetime,
        "test_run_datetime": timestamp,
        "test_run_type": run_type,
        "test_run_status": tests_exit_code,
        "composer_env": compser_env,
        "task_group_id": task_group_id,
        "gcs_folder": gcs_folder,
        "authed_url_logfile": authed_gcs_logfile_link,
        "authed_url_html": authed_gcs_html_link,
        "authed_url_xmlresults": authed_gcs_xmlresults_link,
        "authed_url_dockerlog": authed_gcs_dockerlog_link,
    })
  else:
    for testsuite in root.findall("testsuite"):
      hostname = testsuite.get("hostname")
      timestamp_str = testsuite.get("timestamp")
      timestamp = str(datetime.datetime.fromisoformat(timestamp_str))
      tests = int(testsuite.get("tests"))
      errors = int(testsuite.get("errors"))
      failures = int(testsuite.get("failures"))
      skipped = int(testsuite.get("skipped"))

      rows_to_insert.append({
          "execution_datetime": exec_iso8601_datetime,
          "test_run_datetime": timestamp,
          "test_run_type": run_type,
          "test_run_status": tests_exit_code,
          "composer_env": compser_env,
          "task_group_id": task_group_id,
          "hostname": hostname,
          "tests": tests,
          "errors": errors,
          "failures": failures,
          "skipped": skipped,
          "gcs_folder": gcs_folder,
          "authed_url_logfile": authed_gcs_logfile_link,
          "authed_url_html": authed_gcs_html_link,
          "authed_url_xmlresults": authed_gcs_xmlresults_link,
          "authed_url_dockerlog": authed_gcs_dockerlog_link,
      })

  if rows_to_insert:
    errors = client.insert_rows_json(table_ref, rows_to_insert)
    if errors:
      print(f"Errors inserting rows: {errors}")
      return False
    else:
      print(f"Successfully inserted {len(rows_to_insert)} rows into BigQuery.")
      return True
  else:
    print("No data to insert.")
    return True


def parse_xml_and_insert_details_bq(
    xml_string, project_id, dataset_id, table_id
) -> bool:
  """
  Parses low level XML test results and inserts them into a BigQuery table.

  Args:
      xml_string: The XML string containing test results.
      project_id: Your Google Cloud Project ID.
      dataset_id: The BigQuery dataset ID.
      table_id: The BigQuery table ID.
  """
  try:
    root = ET.fromstring(xml_string)
  except ET.ParseError as e:
    print(f"Error parsing XML: {e}")
    return False

  client = bigquery.Client(project=project_id)
  dataset_ref = client.dataset(dataset_id)

  # Define the table schema
  schema = [
      bigquery.SchemaField("test_name", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("tests", "INTEGER", mode="NULLABLE"),
      bigquery.SchemaField("time", "FLOAT", mode="NULLABLE"),
      bigquery.SchemaField("timestamp", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("hostname", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("classname", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("testcase_name", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("testcase_time", "FLOAT", mode="NULLABLE"),
      bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
      bigquery.SchemaField("message_content", "STRING", mode="NULLABLE"),
  ]

  print(f"{client.list_tables(dataset_ref)}")
  table_ref = bigquery.Table(
      f"{project_id}.{dataset_id}.{table_id}", schema=schema
  )
  if not check_bigquery_table_exists(project_id, dataset_id, table_id):
    try:
      table_ref = client.create_table(table_ref)
      print(
          f"Created table {table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"
      )
    except Exception as e:
      print(f"Got Exception {e}")
      print(f"{pprint.pprint(e)}")
      print(
          f"The exception class is: {e.__class__.__module__}.{e.__class__.__name__}"
      )
      return False

  rows_to_insert = []
  for testsuite in root.findall("testsuite"):
    test_name = testsuite.get("name")
    errors = int(testsuite.get("errors"))
    tests = int(testsuite.get("tests"))
    time = float(testsuite.get("time"))
    timestamp_str = testsuite.get("timestamp")
    hostname = testsuite.get("hostname")
    timestamp = str(datetime.datetime.fromisoformat(timestamp_str))

    for testcase in testsuite.findall("testcase"):
      classname = testcase.get("classname")
      testcase_name = testcase.get("name")
      testcase_time = float(testcase.get("time"))

      status = "pass"
      message_content = ""

      skipped_tag = testcase.find("skipped")
      failure_tag = testcase.find("failure")
      error_tag = testcase.find("error")

      if skipped_tag is not None:
        status = "skip"
        message_content = skipped_tag.get("message", "")
        if skipped_tag.text:
          message_content += "\n" + skipped_tag.text.strip()
      elif failure_tag is not None:
        status = "fail"
        message_content = failure_tag.get("message", "")
        if failure_tag.text:
          message_content += "\n" + failure_tag.text.strip()
      elif error_tag is not None:
        status = "error"
        message_content = error_tag.get("message", "")
        if error_tag.text:
          message_content += "\n" + error_tag.text.strip()

      rows_to_insert.append({
          "test_name": test_name,
          "tests": tests,
          "time": time,
          "timestamp": timestamp,
          "hostname": hostname,
          "classname": classname,
          "testcase_name": testcase_name,
          "testcase_time": testcase_time,
          "status": status,
          "message_content": message_content,
      })

  # Insert 'block_size' rows into BQ at a time to avoid trying to send a single update that is too big
  block_size = 1000
  print(
      f"Insering {len(rows_to_insert)} rows of test results, in chunks of {block_size} rows"
  )
  if rows_to_insert:
    for block in range(0, len(rows_to_insert), block_size):
      block_end = (
          block + block_size
          if block + block_size < len(rows_to_insert)
          else len(rows_to_insert)
      )
      print(f"Inserting rows [{block}:{block_end}].")
      block_data = rows_to_insert[block:block_end]
      errors = client.insert_rows_json(table_ref, block_data)
      if errors:
        print(f"Errors inserting rows [{block}:{block + block_size}]: {errors}")
        return False
    print(f"Successfully inserted {len(rows_to_insert)} rows into BigQuery.")
  else:
    print("No data to insert.")

  return True


# Static global configuration options for BQ dataset/tables
# TODO - change static configuration to options on command line etc
project_id = "cloud-ml-auto-solutions"
dataset_id = "xlml_bite_testresults"
table_test_details_id = "bite_unittest_details"
table_test_summary_id = "bite_unittest_summary"


#
# Main function
#
if __name__ == "__main__":
  if len(sys.argv) < 8:
    print(
        f"usage: {sys.argv[0]} <tests_exit_code> <xml_input_file> <run_type> <run_date_iso8601> <composer_env> <task_id> <gcs_log_file_folder> <stdout_log_file> <dockerlog_file> <html_file>"
    )
    sys.exit(1)

  tests_exit_code = sys.argv[1]
  xmlfile = sys.argv[2]
  run_type = sys.argv[3]
  iso8601_datetime = sys.argv[4]
  compser_env = sys.argv[5]
  task_id = sys.argv[6]
  gcs_bucket_folder = sys.argv[7]
  stdout_filename = sys.argv[8]
  dockerlog_filename = sys.argv[9]
  html_filename = sys.argv[10]

  chars_to_remove = "-:."
  filesafe_datetime = iso8601_datetime
  for ch in chars_to_remove:
    filesafe_datetime = filesafe_datetime.replace(ch, "")

  print(
      f"Exporting XML test results in {xmlfile} for run to BigQuery Dataset {project_id}.{dataset_id}"
  )

  process_xml_data = True

  try:
    with open(xmlfile, "r") as xmlin:
      xmldata = xmlin.read()
  except Exception as e:
    print("Error reading XML file:", e)
    process_xml_data = False
    traceback.print_exc(file=sys.stdout)

  # Still try to add a summary entry in BQ even if we couldn't read the XML output / the
  # XML tests output doesn't exist i.e. the pytests failed to complete their run
  if not process_xml_data:
    xmldata = None
  if not parse_xml_and_insert_summary_bq(
      tests_exit_code,
      xmldata,
      project_id,
      dataset_id,
      table_test_summary_id,
      run_type,
      iso8601_datetime,
      compser_env,
      task_id,
      gcs_bucket_folder,
      stdout_filename,
      dockerlog_filename,
      html_filename,
  ):
    print("Error failed to add summary entry to BQ")
    sys.exit(1)

  if process_xml_data:
    if not parse_xml_and_insert_details_bq(
        xmldata, project_id, dataset_id, table_test_details_id
    ):
      print("Error failed to add tests details information to BQ")
      sys.exit(1)

  sys.exit(0)
