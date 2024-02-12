resource "google_container_cluster" "gpu-uc1" {
  name     = "wcromar-test-cluster"
  project  = "cloud-ml-auto-solutions"
  location = "us-central1"

  release_channel {
    channel = "RAPID"
  }

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "primary" {
  name       = "primary-pool"
  project  = google_container_cluster.gpu-uc1.project
  location   = google_container_cluster.gpu-uc1.location
  cluster    = google_container_cluster.gpu-uc1.name
  node_count = 1

  management {
    auto_repair = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = true
    machine_type = "e2-medium"

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    # TODO: custom service account?
    # service_account = google_service_account.default.email
    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}


resource "google_container_node_pool" "nvidia-v100x2" {
  name       = "nvidia-v100x2-pool"
  project  = google_container_cluster.gpu-uc1.project
  location   = google_container_cluster.gpu-uc1.location
  cluster    = google_container_cluster.gpu-uc1.name
  node_count = 3

  node_locations = [
    "us-central1-b"
  ]

  management {
    auto_repair = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = true
    machine_type = "n1-highmem-16"

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    # TODO: custom service account?
    # service_account = google_service_account.default.email
    oauth_scopes    = [
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
