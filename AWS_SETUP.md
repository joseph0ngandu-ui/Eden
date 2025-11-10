# AWS Setup Guide for Eden Trading Bot

## AWS Credentials Required

To complete the Eden backend deployment to AWS, you'll need:

1. **AWS Access Key ID**
2. **AWS Secret Access Key**
3. **Default Region**: `us-east-1` (recommended)
4. **Default Output Format**: `json` (recommended)

## Getting AWS Credentials

### Option 1: If You Don't Have AWS Account
1. Go to [aws.amazon.com](https://aws.amazon.com/)
2. Click "Create an AWS account"
3. Follow the registration process

### Option 2: If You Have AWS Account
1. Log in to the AWS Management Console
2. Click on your username in the top-right corner
3. Select "My Security Credentials"
4. Click "Access keys (access key ID and secret access key)"
5. Click "Create access key"
6. Select "Command Line Interface (CLI)" as the use case
7. Follow the prompts and save your credentials securely

## Running AWS Configure

Once you have your AWS credentials, run:

```powershell
# Create alias for AWS CLI (only needed once)
Set-Alias -Name "aws" -Value "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

# Configure AWS
aws configure
```

You'll be prompted for:
1. **AWS Access Key ID**: `[Enter your access key]`
2. **AWS Secret Access Key**: `[Enter your secret key]` (will be hidden)
3. **Default region name**: `us-east-1`
4. **Default output format**: `json`

## Creating Eden Deployment IAM User (Recommended)

For better security, create a dedicated IAM user with minimal permissions:

1. Log into the AWS Console
2. Go to IAM → Users → Create User
3. Create a user named "eden-deployer"
4. In the next step, select "Attach policies directly"
5. Add these policies:
   - `AmazonEC2FullAccess` (for VPC management)
   - `AmazonECSFullAccess` (for container services)
   - `AmazonRDSFullAccess` (for database)
   - `AmazonAPIGatewayFullAccess` (for API Gateway)
   - `AmazonEC2ContainerRegistryFullAccess` (for ECR)
   - `IAMFullAccess` (for role management)
6. Proceed without AWS Console access
7. Create access key for CLI access
8. Note down the access key ID and secret access key

## Once Configured

After running `aws configure`, you can verify the setup:

```powershell
aws sts get-caller-identity
```

This should display your AWS account information, confirming the configuration was successful.

## Ready for Deployment

Once AWS CLI is configured, you'll be ready to:

1. Create an ECR repository
2. Push Docker image to ECR
3. Deploy CloudFormation stack
4. Get the API Gateway URL
5. Update your iOS app with the production URL

## Need Help?

If you encounter any issues:
- Check that your AWS credentials are correct
- Verify the region matches your intended deployment region
- Ensure the IAM user has sufficient permissions (as listed above)