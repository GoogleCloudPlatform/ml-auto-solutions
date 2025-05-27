resource "google_composer_environment" "example_environment" {
  provider = google-beta
  name     = var.environment_name
  region = var.region

  config {
    environment_size = "ENVIRONMENT_SIZE_MEDIUM"
    software_config {
      image_version = "composer-2.13.1-airflow-2.10.5"
      airflow_config_overrides = {
        core-allowed_deserialization_classes_regexp = ".*"
        core-dags_are_paused_at_creation = false
        scheduler-min_file_process_interval  = "120"
      }
      # Note: keep this in sync with .github/requirements.txt
      pypi_packages = {
        apache-airflow-providers-sendgrid = ""
        fabric                            = ""
        google-cloud-tpu                  = ">=1.16.0"
        jsonlines                         = ""
        ray                               = "[default]"
        dacite                            = ""
        # These packages are already in the default composer environment.
        # See https://cloud.google.com/composer/docs/concepts/versioning/composer-versions
        # google-cloud-bigquery             = ""
        # google-cloud-storage              = ""
        # google-cloud-container            = ""
        # tensorflow-cpu                    = ""
        # kubernetes                        = ""
        # pyarrow                           = ""
      }
    }

    workloads_config {
      scheduler {
        cpu        = 28
        memory_gb  = 80
        storage_gb = 10
        count      = 2
      }
      web_server {
        cpu        = 2
        memory_gb  = 8
        storage_gb = 10
      }
      worker {
        cpu        = 8
        memory_gb  = 48
        storage_gb = 10
        min_count  = 1
        max_count  = 100
      }
    }

    node_config {
      service_account = var.service_account
    }
  }
}
