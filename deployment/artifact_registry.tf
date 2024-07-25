resource "google_artifact_registry_repository" "private-xlml-index" {
  project       = var.project_config.project_name
  location      = "us-central1"
  repository_id = "xlml-private"
  description   = "Packaged `xlml` wheels"
  format        = "python"
}
