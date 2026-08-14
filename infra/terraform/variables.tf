variable "project_name" {
  description = "Project name used to prefix resource names."
  type        = string
  default     = "documind-ai"
}

variable "environment" {
  description = "Deployment environment (dev, stage, prod)."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS cluster's OIDC provider, used for IRSA (IAM Roles for Service Accounts)."
  type        = string
  default     = "arn:aws:iam::000000000000:oidc-provider/REPLACE_ME"
}
