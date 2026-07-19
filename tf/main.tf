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


data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
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

# allow unauthenticated access to the service 
resource "google_cloud_run_service_iam_policy" "noauth2" {
  location    = google_cloud_run_service.fast-api.location
  project     = google_cloud_run_service.fast-api.project
  service     = google_cloud_run_service.fast-api.name

  policy_data = data.google_iam_policy.noauth.policy_data
}

# jabras@Jacobs-MacBook-Pro TF % terraform apply use.plan

# google_service_account.gcs_sa: Creating...
# google_cloud_run_service.fast-api: Creating...
# google_service_account.gcs_sa: Creation complete after 1s [id=projects/seventh-history-374820/serviceAccounts/gcs-sa@seventh-history-374820.iam.gserviceaccount.com]
# google_cloud_run_service.fast-api: Still creating... [10s elapsed]
# google_cloud_run_service.fast-api: Still creating... [20s elapsed]
# google_cloud_run_service.fast-api: Still creating... [30s elapsed]
# google_cloud_run_service.fast-api: Still creating... [40s elapsed]
# google_cloud_run_service.fast-api: Still creating... [50s elapsed]

# Error: Error waiting to create Service: resource is in failed state "Ready:False", message: Revision 'tokyo-run-00001-fct' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=8000 environment variable. Logs for this revision might contain more information.

# Logs URL: https://console.cloud.google.com/logs/viewer?project=seventh-history-374820&resource=cloud_run_revision/service_name/tokyo-run/revision_name/tokyo-run-00001-fct&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22tokyo-run%22%0Aresource.labels.revision_name%3D%22tokyo-run-00001-fct%22 
# For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start

#   on main.tf line 56, in resource "google_cloud_run_service" "fast-api":
#   56: resource "google_cloud_run_service" "fast-api" {
