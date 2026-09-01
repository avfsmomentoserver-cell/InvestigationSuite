variable "aws_region" { type = string, default = "us-east-1" }
variable "project_name" { type = string, default = "forecasting-rounds" }
variable "instance_type" { type = string, default = "t3.large" }
variable "key_name" { type = string }
variable "public_key_path" { type = string, default = "~/.ssh/id_rsa.pub" }
variable "vpc_id" { type = string, default = "" }
variable "subnet_id" { type = string, default = "" }
variable "allowed_ssh_cidr" { type = string, default = "0.0.0.0/0" }
variable "app_port" { type = number, default = 8000 }
