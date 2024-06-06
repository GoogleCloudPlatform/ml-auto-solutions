resource "google_container_cluster" "gpu-uc1" {
  name     = var.cluster_name
  project  = var.project_id
  location = var.location

  release_channel {
    channel = "RAPID"
  }

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "primary" {
  name       = var.node_pool_name
  project    = var.project_id
  location   = var.location
  cluster    = google_container_cluster.gpu-uc1.name
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = true
    machine_type = "e2-medium"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "google_container_node_pool" "nvidia-v100x2" {
  name       = var.gpu_node_pool_name
  project    = var.project_id
  location   = var.location
  cluster    = google_container_cluster.gpu-uc1.name

  autoscaling {
    min_node_count = 2
    max_node_count = 6
  }

  node_locations = [
    "us-central1-b"
  ]

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = true
    machine_type = "n1-highmem-16"
    disk_size_gb = 500
    disk_type = "pd-balanced"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    guest_accelerator {
      type  = "nvidia-tesla-v100"
      count = 2
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
  }
}

data "google_client_config" "provider" {}

provider "kubernetes" {
  host  = "https://${google_container_cluster.gpu-uc1.endpoint}"
  token = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.gpu-uc1.master_auth[0].cluster_ca_certificate,
  )
}

resource "kubernetes_service" "example" {
  metadata {
    name = "headless-svc"
  }
  spec {
    selector = {
      headless-svc = "true"
    }
    cluster_ip = "None"
  }
}
