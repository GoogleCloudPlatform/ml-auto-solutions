provider "google-beta" {
  project = var.project_config.project_name
  region  = var.project_config.project_region
}

terraform {
  backend "gcs" {
    bucket = "us-central1-akshu-test-225d2657-bucket"
    prefix = "terraform/state"
  }
}

