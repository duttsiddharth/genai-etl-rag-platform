# DocuMind AI — AWS infrastructure (reference implementation)
# Persona: MLOps/DevOps Engineer
# JD requirement covered: cloud platform collaboration (AWS) + MLOps/DevOps
# lifecycle management.
#
# NOTE: This module is structurally complete and `terraform validate`-able,
# but is not `apply`-ed against a live AWS account in this reference
# repository (no cloud credentials are provisioned in this workspace).
# A remote state backend is declared but commented out — uncomment and
# point at your own S3/DynamoDB backend before running in a real account.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # backend "s3" {
  #   bucket         = "documind-ai-terraform-state"
  #   key            = "documind-ai/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "documind-ai-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
}

# --- S3 bucket: raw documents + ingestion manifests ---
resource "aws_s3_bucket" "raw_docs" {
  bucket = "${var.project_name}-${var.environment}-raw-docs"
}

resource "aws_s3_bucket_versioning" "raw_docs" {
  bucket = aws_s3_bucket.raw_docs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_docs" {
  bucket = aws_s3_bucket.raw_docs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "raw_docs" {
  bucket                  = aws_s3_bucket.raw_docs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- ECR repository: container images ---
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# --- CloudWatch log group ---
resource "aws_cloudwatch_log_group" "api" {
  name              = "/documind-ai/${var.environment}/api"
  retention_in_days = 30
}

# --- IAM role for the workload (least privilege) ---
data "aws_iam_policy_document" "workload_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"
    principals {
      type        = "Federated"
      identifiers = [var.eks_oidc_provider_arn]
    }
  }
}

resource "aws_iam_role" "workload" {
  name               = "${var.project_name}-${var.environment}-workload"
  assume_role_policy = data.aws_iam_policy_document.workload_assume_role.json
}

data "aws_iam_policy_document" "workload_permissions" {
  statement {
    sid    = "S3ProjectPrefixOnly"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.raw_docs.arn,
      "${aws_s3_bucket.raw_docs.arn}/*",
    ]
  }

  statement {
    sid    = "BedrockInvokeOnly"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "Logs"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.api.arn}:*"]
  }
}

resource "aws_iam_role_policy" "workload" {
  name   = "${var.project_name}-${var.environment}-workload-policy"
  role   = aws_iam_role.workload.id
  policy = data.aws_iam_policy_document.workload_permissions.json
}
