
variable "project_id" {
  type    = string
  default = "cloud-ml-auto-solutions"
}

variable "location" {
  type    = string
  default = "us-central1"
}

variable "cluster_name" {
  type    = string
  default = "gpu-uc1"
}

variable "node_pool_name" {
  type    = string
  default = "primary-pool"
}

variable "gpu_node_pool_name" {
  type    = string
  default = "nvidia-v100x2-pool"
}
