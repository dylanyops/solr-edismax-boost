variable "aws_region" {
  default = "us-east-1"
}

variable "cluster_name" {
  default = "ml-platform"
}

variable "db_username" {
  default = "mlflow"
}

variable "db_password" {
  sensitive = true
}