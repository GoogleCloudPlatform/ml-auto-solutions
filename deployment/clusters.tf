resource "google_container_cluster" "gpu-uc1" {
  name     = "gpu-uc1"
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

  autoscaling {
    min_node_count = 2
    max_node_count = 6
  }

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
    disk_size_gb = 500
    disk_type = "pd-balanced"

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

data "google_client_config" "provider" {}

provider "kubernetes" {
  host  = "https://${google_container_cluster.gpu-uc1.endpoint}"
  token = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.gpu-uc1.master_auth[0].cluster_ca_certificate,
  )
}

// Headless service required for service discovery within a Job.
// Pods will be addressable as `hostname.headless-svc`
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

resource "google_container_cluster" "benchmarking-gpu-uc1" {
  name     = "benchmarking-gpu-uc1"
  project  = "cloud-ml-benchmarking"
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

resource "google_container_node_pool" "benchmarking-primary" {
  name       = "benchmarking-primary-pool"
  project  = google_container_cluster.benchmarking-gpu-uc1.project
  location   = google_container_cluster.benchmarking-gpu-uc1.location
  cluster    = google_container_cluster.benchmarking-gpu-uc1.name
  node_count = 1

  management {
    auto_repair = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = true
    machine_type = "e2-medium"

    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

resource "google_container_node_pool" "nvidia-h100x8" {
  name       = "nvidia-h100x8-pool"
  project  = google_container_cluster.benchmarking-gpu-uc1.project
  location   = google_container_cluster.benchmarking-gpu-uc1.location
  cluster    = google_container_cluster.benchmarking-gpu-uc1.name
  initial_node_count = 4

  autoscaling {
    location_policy = "ANY"
    total_min_node_count = 2
    total_max_node_count = 4
  }

  node_locations = [
    "us-central1-a"
  ]

  management {
    auto_repair = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = false
    machine_type = "a3-highgpu-8g"
    disk_size_gb = 500
    disk_type = "pd-ssd"
    local_ssd_count = 0
    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    guest_accelerator {
      type  = "nvidia-h100-80gb"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
  }

  lifecycle {
    ignore_changes = [
      # Ignore changes to local_ssd_count. Reason mentioned in
      # https://github.com/hashicorp/terraform-provider-google/issues/17068
      node_config["local_ssd_count"],
    ]
  }
}

resource "google_container_node_pool" "nvidia-a100x1" {
  name       = "nvidia-a100x1-pool"
  project  = google_container_cluster.benchmarking-gpu-uc1.project
  location   = google_container_cluster.benchmarking-gpu-uc1.location
  cluster    = google_container_cluster.benchmarking-gpu-uc1.name
  initial_node_count = 6

  autoscaling {
    location_policy = "ANY"
    total_min_node_count = 6
    total_max_node_count = 12
  }

  node_locations = [
    "us-central1-a"
  ]

  management {
    auto_repair = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = true
    machine_type = "a2-highgpu-1g"
    disk_size_gb = 500
    disk_type = "pd-balanced"

    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    guest_accelerator {
      type  = "nvidia-tesla-a100"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
  }
}

resource "google_container_node_pool" "nvidia-l4x1" {
  name       = "nvidia-l4x1-pool"
  project  = google_container_cluster.benchmarking-gpu-uc1.project
  location   = google_container_cluster.benchmarking-gpu-uc1.location
  cluster    = google_container_cluster.benchmarking-gpu-uc1.name
  initial_node_count = 6

  autoscaling {
    location_policy = "ANY"
    total_min_node_count = 4
    total_max_node_count = 8
  }

  node_locations = [
    "us-central1-a"
  ]

  management {
    auto_repair = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = false
    machine_type = "g2-standard-16"
    disk_size_gb = 500
    disk_type = "pd-balanced"

    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
  }
}

resource "google_container_node_pool" "nvidia-v100x2-bm" {
  name       = "nvidia-v100x2-pool-bm"
  project  = google_container_cluster.benchmarking-gpu-uc1.project
  location   = google_container_cluster.benchmarking-gpu-uc1.location
  cluster    = google_container_cluster.benchmarking-gpu-uc1.name
  initial_node_count = 8

  autoscaling {
    location_policy = "ANY"
    total_min_node_count = 6
    total_max_node_count = 12
  }

  node_locations = [
    "us-central1-c"
  ]

  management {
    auto_repair = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = false
    machine_type = "n1-standard-16"
    disk_size_gb = 500
    disk_type = "pd-balanced"

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
