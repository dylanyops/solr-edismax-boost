resource "aws_db_instance" "mlflow" {
  identifier        = "mlflow-db"
  engine            = "postgres"
  instance_class    = "db.t3.micro"
  allocated_storage = 20

  username = var.db_username
  password = var.db_password

  skip_final_snapshot = true
  publicly_accessible = false
}