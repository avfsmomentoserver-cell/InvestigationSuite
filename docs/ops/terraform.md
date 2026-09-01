# Ops: Terraform & Cloud-init

Instructions to run terraform are here:
- Create terraform/terraform.tfvars with your values (key_name, public_key_path, vpc_id, subnet_id if needed)
- Run:
  terraform init
  terraform plan -var-file="terraform.tfvars"
  terraform apply -var-file="terraform.tfvars"
Be careful: this will create EC2 resources in your AWS account.
