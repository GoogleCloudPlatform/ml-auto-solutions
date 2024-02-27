# # Copyright 2024 Google LLC
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """A DAG to run end-to-end MaxText tests."""


# import datetime
# from airflow import models
# from dags import composer_env
# from dags.vm_resource import TpuVersion, Zone
# from dags.multipod.configs import maxtext_gke_config, common
# from dags.multipod.configs.common import SetupMode, Platform


# # Run once a day at 4 am UTC (8 pm PST)
# SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


# with models.DAG(
#     dag_id="maxtext_convergence",
#     schedule=SCHEDULED_TIME,
#     tags=["multipod_team", "maxtext", "stable", "nightly"],
#     start_date=datetime.datetime(2024, 1, 19),
#     catchup=False,
# ) as dag:
#   test_name_prefix = "maxtext"
#   test_models = {
#       "llama2": ["test_llama2", "llama_finetuning_test"],
#       "mistral": ["test_mistral"],
#       "gamma": ["test_gamma"],
#       "gpt3": ["test_gpt3"],
#   }
#   test_modes = [SetupMode.STABLE, SetupMode.NIGHTLY]
  
#   # bf16 convergence test
#   base_output_directory = "gs://maxtext-experiments-multipod"
#   dataset_path = "gs://max-datasets-rogue"

#   setup_command = common.setup_maxtext(SetupMode.STABLE, Platform.GKE)
#   run_command = f"{setup_command} && bash MaxText/end_to_end/test_convergence.py OUTPUT_PATH={base_output_directory} DATASET={dataset_path}"
#   maxtext_gke_config.get_maxtext_xpk_config(
#     tpu_version=TpuVersion.V4,
#     tpu_cores=128,
#     tpu_zone=Zone.US_CENTRAL2_B.value,
#     test_name="maxtext-bf16-convergence"
#     project_name=Project.TPU_PROD_ENV_MULTIPOD.value,
#     cluster_name=ClusterName.V4_128_MULTISLICE_CLUSTER.value,
#     docker_image=DockerImage.XPK_JAX_TEST.value,
#     time_out_in_min=600,
#     run_model_cmds=run_command,
#     num_slices=1).run()
