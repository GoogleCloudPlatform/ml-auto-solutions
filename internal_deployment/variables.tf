variable "project_id" {
  type = string
  description = "The project ID. E.g., cloud-ml-auto-solutions"
}

variable "region" {
  type = string
  description = "The region your Cloud Composer Env will be created in. E.g., us-east1."
}

variable "env_name" {
  type = string
  description = "The name of the Cloud Composer Env. Something like <your_ldap>-test"
}
