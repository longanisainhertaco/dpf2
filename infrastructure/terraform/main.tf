# Terraform configuration for deploying dpf2 on AWS Batch
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.5.0"
}

provider "aws" {
  region = var.region
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Minimal Batch compute environment
resource "aws_batch_compute_environment" "dpf2" {
  compute_environment_name = "dpf2"
  service_role             = aws_iam_role.batch_service.arn
  type                     = "MANAGED"

  compute_resources {
    max_vcpus = 16
    type      = "EC2"
    subnets   = []
    security_group_ids = []
    instance_types = ["m5.large"]
  }
}

resource "aws_iam_role" "batch_service" {
  name = "dpf2-batch-service-role"
  assume_role_policy = data.aws_iam_policy_document.batch_assume.json
}

data "aws_iam_policy_document" "batch_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

# Job queue and definition would be added here.
