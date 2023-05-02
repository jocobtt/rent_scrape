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
# resource "google_service_account" "gcs_sa" {
#   account_id   = "gcs-sa"
#   display_name = "GCS SA"
# }


# # create cloud storage bucket for storing the model states and other artifacts
# resource "google_storage_bucket" "gcs_bucket" {
#   name          = "bby-gcr-namez-bucket"
#   location      = var.location
#   storage_class = "REGIONAL" # what other options do I have? 
#   # secure the bucket 
#   uniform_bucket_level_access = true
#   service_account_email       = google_service_account.gcs_sa.email
  
# }


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
        ports {
          name = "http1"
          container_port = 8000 
        }
        resources {
          limits = {
            cpu    = "1000m"
            memory = "512M"
          }
        }
      }
      # the service uses this SA to call other Google Cloud APIs
      # service_account_name = myservice_runtime_sa
    }

    metadata {
      annotations = {
        # Limit scale up to prevent any cost blow outs!
        "autoscaling.knative.dev/maxScale" = var.auto_scale
        # all egress from the service should go through the VPC Connector
        #"run.googleapis.com/vpc-access-egress" = "all-traffic"
      }
    }
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