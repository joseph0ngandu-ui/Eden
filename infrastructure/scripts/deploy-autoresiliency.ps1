# Eden Auto-Resilient Infrastructure Deployment Script
# Deploys Lambda functions, EventBridge rules, and CloudWatch alarms

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "prod",
    
    [Parameter(Mandatory=$true)]
    [string]$APIGatewayId,
    
    [Parameter(Mandatory=$true)]
    [string]$APIGatewayIntegrationId,
    
    [Parameter(Mandatory=$true)]
    [string]$APIGatewayURL,
    
    [Parameter(Mandatory=$false)]
    [string]$NotificationEmail = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-east-1",
    
    [Parameter(Mandatory=$false)]
    [string]$StackName = "eden-autoresiliency"
)

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Eden Auto-Resilient Infrastructure Deployment" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Verify AWS CLI is available
Write-Host "✓ Checking AWS CLI..." -ForegroundColor Yellow
try {
    $awsVersion = aws --version 2>&1
    Write-Host "  AWS CLI: $awsVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ AWS CLI not found. Please install AWS CLI first." -ForegroundColor Red
    exit 1
}

# Verify AWS credentials
Write-Host "✓ Verifying AWS credentials..." -ForegroundColor Yellow
try {
    $identity = aws sts get-caller-identity --output json --no-cli-pager | ConvertFrom-Json
    Write-Host "  Account: $($identity.Account)" -ForegroundColor Green
    Write-Host "  User: $($identity.Arn)" -ForegroundColor Green
} catch {
    Write-Host "✗ AWS credentials not configured. Run 'aws configure' first." -ForegroundColor Red
    exit 1
}

# Get ECS cluster and service info
Write-Host ""
Write-Host "✓ Fetching ECS cluster information..." -ForegroundColor Yellow
try {
    $clusters = aws ecs list-clusters --region $Region --output json --no-cli-pager | ConvertFrom-Json
    $clusterName = "$Environment-eden-cluster"
    Write-Host "  Cluster: $clusterName" -ForegroundColor Green
    
    $serviceName = "$Environment-eden-service"
    Write-Host "  Service: $serviceName" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to fetch ECS information" -ForegroundColor Red
    exit 1
}

# Prepare deployment parameters
$parameters = @(
    "ParameterKey=EnvironmentName,ParameterValue=$Environment",
    "ParameterKey=ECSClusterName,ParameterValue=$clusterName",
    "ParameterKey=ECSServiceName,ParameterValue=$serviceName",
    "ParameterKey=APIGatewayId,ParameterValue=$APIGatewayId",
    "ParameterKey=APIGatewayIntegrationId,ParameterValue=$APIGatewayIntegrationId",
    "ParameterKey=APIGatewayURL,ParameterValue=$APIGatewayURL"
)

if ($NotificationEmail -ne "") {
    $parameters += "ParameterKey=NotificationEmail,ParameterValue=$NotificationEmail"
}

$parametersString = $parameters -join " "

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Deployment Configuration" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Environment:       $Environment" -ForegroundColor White
Write-Host "  Region:            $Region" -ForegroundColor White
Write-Host "  Stack Name:        $StackName" -ForegroundColor White
Write-Host "  ECS Cluster:       $clusterName" -ForegroundColor White
Write-Host "  ECS Service:       $serviceName" -ForegroundColor White
Write-Host "  API Gateway ID:    $APIGatewayId" -ForegroundColor White
Write-Host "  Integration ID:    $APIGatewayIntegrationId" -ForegroundColor White
Write-Host "  API Gateway URL:   $APIGatewayURL" -ForegroundColor White
if ($NotificationEmail -ne "") {
    Write-Host "  Notification Email: $NotificationEmail" -ForegroundColor White
}
Write-Host ""

# Confirm deployment
$confirm = Read-Host "Deploy auto-resilient infrastructure? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Deployment cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "✓ Deploying CloudFormation stack..." -ForegroundColor Yellow
Write-Host ""

# Deploy CloudFormation stack
try {
    $templatePath = Join-Path $PSScriptRoot "..\aws-autoresiliency.yml"
    
    # Check if stack exists
    $stackExists = $false
    try {
        aws cloudformation describe-stacks `
            --stack-name $StackName `
            --region $Region `
            --no-cli-pager 2>&1 | Out-Null
        $stackExists = $true
        Write-Host "  Stack exists, updating..." -ForegroundColor Cyan
    } catch {
        Write-Host "  Creating new stack..." -ForegroundColor Cyan
    }
    
    if ($stackExists) {
        # Update stack
        aws cloudformation update-stack `
            --stack-name $StackName `
            --template-body "file://$templatePath" `
            --parameters $parametersString `
            --capabilities CAPABILITY_NAMED_IAM `
            --region $Region `
            --no-cli-pager
        
        $operation = "update"
    } else {
        # Create stack
        aws cloudformation create-stack `
            --stack-name $StackName `
            --template-body "file://$templatePath" `
            --parameters $parametersString `
            --capabilities CAPABILITY_NAMED_IAM `
            --region $Region `
            --no-cli-pager
        
        $operation = "create"
    }
    
    Write-Host "  Stack $operation initiated..." -ForegroundColor Green
    Write-Host ""
    Write-Host "✓ Waiting for stack $operation to complete..." -ForegroundColor Yellow
    Write-Host "  (This may take 3-5 minutes)" -ForegroundColor Gray
    
    # Wait for stack operation to complete
    aws cloudformation wait "stack-${operation}-complete" `
        --stack-name $StackName `
        --region $Region
    
    Write-Host ""
    Write-Host "✅ Stack deployed successfully!" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "✗ Stack deployment failed: $_" -ForegroundColor Red
    exit 1
}

# Get stack outputs
Write-Host ""
Write-Host "✓ Retrieving stack outputs..." -ForegroundColor Yellow
try {
    $stackInfo = aws cloudformation describe-stacks `
        --stack-name $StackName `
        --region $Region `
        --output json `
        --no-cli-pager | ConvertFrom-Json
    
    $outputs = $stackInfo.Stacks[0].Outputs
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Deployment Complete!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    foreach ($output in $outputs) {
        Write-Host "  $($output.OutputKey):" -ForegroundColor White
        Write-Host "    $($output.OutputValue)" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "✅ Auto-Resilient Infrastructure Deployed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Features Enabled:" -ForegroundColor Cyan
    Write-Host "  ✓ Automatic API Gateway IP updates on ECS task changes" -ForegroundColor White
    Write-Host "  ✓ Health monitoring with CloudWatch alarms" -ForegroundColor White
    Write-Host "  ✓ Automatic ECS task restart on failures" -ForegroundColor White
    Write-Host "  ✓ Scheduled health checks every 5 minutes" -ForegroundColor White
    if ($NotificationEmail -ne "") {
        Write-Host "  ✓ Email notifications to: $NotificationEmail" -ForegroundColor White
    }
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Check CloudWatch Logs for Lambda execution logs" -ForegroundColor White
    Write-Host "  2. Test by stopping an ECS task: aws ecs stop-task --cluster $clusterName --task <task-id>" -ForegroundColor White
    Write-Host "  3. Monitor API Gateway for automatic IP updates" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host "✗ Failed to retrieve stack outputs: $_" -ForegroundColor Red
    exit 1
}

Write-Host "Deployment script completed successfully!" -ForegroundColor Green
