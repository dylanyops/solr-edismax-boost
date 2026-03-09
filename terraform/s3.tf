resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "mlflow-artifacts-${var.cluster_name}"
}

resource "aws_s3_bucket" "feast_offline_store" {
  bucket = "feast-offline-${var.cluster_name}"
}

resource "aws_s3_bucket" "evidently_reports" {
  bucket = "evidently-reports-${var.cluster_name}"
}