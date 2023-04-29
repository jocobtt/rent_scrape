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

variable "app_version" {
  type    = string 
  default = "latest"
}


variable "auto_scale" {
  type    = number 
  default = "3"
}

