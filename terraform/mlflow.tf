resource "kubernetes_namespace" "mlflow" {
  metadata {
    name = "mlflow"
  }
}

resource "kubernetes_deployment" "mlflow" {
  metadata {
    name      = "mlflow"
    namespace = kubernetes_namespace.mlflow.metadata[0].name
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "mlflow"
      }
    }

    template {
      metadata {
        labels = {
          app = "mlflow"
        }
      }

      spec {
        container {
          name  = "mlflow"
          image = "ghcr.io/mlflow/mlflow:latest"

          command = [
            "mlflow",
            "server",
            "--backend-store-uri",
            "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.mlflow.address}:5432/postgres",
            "--default-artifact-root",
            "s3://${aws_s3_bucket.mlflow_artifacts.bucket}",
            "--host",
            "0.0.0.0"
          ]

          port {
            container_port = 5000
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "mlflow" {
  metadata {
    name      = "mlflow"
    namespace = kubernetes_namespace.mlflow.metadata[0].name
  }

  spec {
    selector = {
      app = "mlflow"
    }

    port {
      port        = 5000
      target_port = 5000
    }

    type = "ClusterIP"
  }
}