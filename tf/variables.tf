variable "service_account_file" {
  type    = string
  default = "~/.gcp-sa-key.json"
}

variable "project_id" {
  type        = string
  description = "GCP project ID to deploy into"
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


variable "allow_public_access" {
  type        = bool
  default     = false
  description = "Grant roles/run.invoker to allUsers. Leave false and pass invoker_members unless the API sits behind its own auth (API_KEY)."
}

variable "invoker_members" {
  type        = list(string)
  default     = []
  description = "IAM members allowed to invoke the Cloud Run service, e.g. [\"user:me@example.com\"] or [\"serviceAccount:...\"]"
}

variable "admin_api_key" {
  type        = string
  sensitive   = true
  description = "Value for ADMIN_API_KEY (guards training/promotion/rollback endpoints). Prefer Secret Manager for real deployments."
}

variable "allowed_origins" {
  type        = string
  default     = ""
  description = "Comma-separated CORS origins for the API"
}
