output "service_name" {
  value       = google_cloud_run_service.fast-api.name
  description = "Name of the created service"
}


output "service_url" {
    value       = google_cloud_run_service.fast-api.status[0].url 
    description = "The URL on which the deployed service is available"
}