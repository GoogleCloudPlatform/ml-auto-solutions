# Cloud Composer Deployment for Quick Testing

This page explains how to quickly set up a Cloud Composer 2 Environment via Terraform.

## Prerequisites
* You have a [Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) to use.
* You have installed [gCloud CLI](https://cloud.google.com/sdk/docs/install).

## Step 1 - Install Terraform

```
./setup_terraform.sh
```

## Step 2 - Initialize Terraform

```
terraform init
```

## Step 3 - Create your Cloud Composer 2 Environment

```
terraform apply
```

1. Fill in the fields when prompted.
1. Once you can see the plan, confirm that you are only creating 1 resource, and not destroying anything.
1. Enter `y` to continue.
1. Your Cloud Composer 2 Environment should take around 20 minutes to create.
