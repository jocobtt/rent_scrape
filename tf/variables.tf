variable "service_account_file" {
  type    = string
  default = "~/.cloud_run_sa.json"
}

variable "project_id" {
  type    = string
  default = "seventh-history-374820"
}

variable "location" {
  type    = string
  default = "us-east1"
}

variable "create_bucket" {
  type    = bool
  default = false
}

variable "app_version" {
  type    = string
  default = "latest"
}

variable "auto_scale" {
  type    = number
  default = 10
  description = "Maximum number of instances for auto-scaling"
}

variable "min_scale" {
  type    = number
  default = 1
  description = "Minimum number of instances (0 = scale to zero, 1+ = always warm)"
}

variable "cpu_limit" {
  type    = string
  default = "2000m"
  description = "CPU limit for the container (e.g., 1000m = 1 vCPU, 2000m = 2 vCPU, 4000m = 4 vCPU)"
}

variable "memory_limit" {
  type    = string
  default = "2Gi"
  description = "Memory limit for the container (ML models need more memory). Options: 512Mi, 1Gi, 2Gi, 4Gi, 8Gi"
}

variable "cpu_throttling" {
  type    = bool
  default = false
  description = "Whether to throttle CPU when idle (set to true to save costs)"
}

variable "request_timeout" {
  type    = number
  default = 300
  description = "Request timeout in seconds (important for model predictions)"
}

variable "concurrency" {
  type    = number
  default = 80
  description = "Maximum concurrent requests per instance (Cloud Run default: 80)"
}

variable "mlflow_tracking_uri" {
  type    = string
  default = "file:///tmp/mlruns"
  description = "MLflow tracking URI. Use GCS path for production (e.g., gs://bucket-name/mlruns)"
}

