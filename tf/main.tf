terraform {
  required_version = ">= 0.13"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "< 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "< 5.0"
    }
  }
}

provider "google" {
  credentials = var.service_account_file 
  project     = var.project_id
}

provider "google-beta" {
  credentials = var.service_account_file 
  project     = var.project_id
}

# create a service account to attribute to the bucket 
resource "google_service_account" "gcs_sa" {
  account_id   = "gcs-sa"
  display_name = "GCS SA"
}


# create cloud storage bucket for storing the model states and other artifacts
resource "google_storage_bucket" "gcs_bucket" {
  count         = var.create_bucket ? 1 : 0
  name          = "rent-scraperz-bucket"
  location      = var.location
  storage_class = "REGIONAL"  
  # secure the bucket 
  uniform_bucket_level_access = true
  
}


data "google_iam_policy" "invokers" {
  binding {
    role    = "roles/run.invoker"
    members = var.allow_public_access ? ["allUsers"] : var.invoker_members
  }
}


// fastapi app
resource "google_cloud_run_service" "fast-api" {
  name     = "tokyo-run"

  location = var.location

  template {
    spec {
      containers {
        image = format("gcr.io/%s/tokyo-model-api:%s", var.project_id, var.app_version)

        # Environment variables
        env {
          name  = "PORT"
          value = "8000"
        }

        env {
          name  = "MLFLOW_TRACKING_URI"
          value = var.mlflow_tracking_uri
        }

        env {
          name  = "ADMIN_API_KEY"
          value = var.admin_api_key
        }

        env {
          name  = "ALLOWED_ORIGINS"
          value = var.allowed_origins
        }

        env {
          name  = "PYTHONUNBUFFERED"
          value = "1"
        }

        ports {
          name           = "http1"
          container_port = 8000
        }

        # Health check configuration - Using new Kubernetes-style probes
        startup_probe {
          http_get {
            path = "/startup"
            port = 8000
          }
          initial_delay_seconds = 10
          timeout_seconds       = 5
          period_seconds        = 10
          failure_threshold     = 6  # 60 seconds total for model loading
        }

        liveness_probe {
          http_get {
            path = "/health"
            port = 8000
          }
          initial_delay_seconds = 0  # Start immediately after startup probe passes
          timeout_seconds       = 5
          period_seconds        = 30
          failure_threshold     = 3
        }

        # Readiness probe for traffic routing
        # Cloud Run uses startup + liveness, but we document readiness for k8s
        # For Cloud Run, the startup probe acts as readiness

        resources {
          limits = {
            cpu    = var.cpu_limit
            memory = var.memory_limit
          }
        }
      }

      # Request timeout (important for model predictions)
      timeout_seconds = var.request_timeout

      # Maximum concurrent requests per instance
      container_concurrency = var.concurrency

      # the service uses this SA to call other Google Cloud APIs
      service_account_name = google_service_account.gcs_sa.email
    }

    metadata {
      annotations = {
        # Auto-scaling configuration
        "autoscaling.knative.dev/maxScale" = tostring(var.auto_scale)
        "autoscaling.knative.dev/minScale" = tostring(var.min_scale)

        # Performance optimizations
        "run.googleapis.com/startup-cpu-boost" = "true"
        "run.googleapis.com/cpu-throttling" = tostring(var.cpu_throttling)

        # Execution environment
        "run.googleapis.com/execution-environment" = "gen2"

        # Session affinity for better performance (optional)
        # "run.googleapis.com/sessionAffinity" = "true"

        # all egress from the service should go through the VPC Connector
        #"run.googleapis.com/vpc-access-egress" = "all-traffic"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true

}

# restrict invocation to var.invoker_members (public only if allow_public_access = true)
resource "google_cloud_run_service_iam_policy" "invokers" {
  location    = google_cloud_run_service.fast-api.location
  project     = google_cloud_run_service.fast-api.project
  service     = google_cloud_run_service.fast-api.name

  policy_data = data.google_iam_policy.invokers.policy_data
}
