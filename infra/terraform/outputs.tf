output "s3_bucket_name" {
  value = aws_s3_bucket.raw_docs.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "cloudwatch_log_group" {
  value = aws_cloudwatch_log_group.api.name
}

output "workload_role_arn" {
  value = aws_iam_role.workload.arn
}
