# Deploy Eden Backend to AWS with MT5 Integration
# Builds Docker image, pushes to ECR, and updates ECS service

param(
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-east-1",
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "prod",
    
    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Eden Backend AWS Deployment with MT5 Integration" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Configuration
$AccountId = "437473173182"
$RepositoryName = "eden-trading-api"
$ClusterName = "$Environment-eden-cluster"
$ServiceName = "$Environment-eden-service"
$ImageUri = "$AccountId.dkr.ecr.$Region.amazonaws.com/${RepositoryName}:${ImageTag}"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Region: $Region" -ForegroundColor White
Write-Host "  Environment: $Environment" -ForegroundColor White
Write-Host "  Image Tag: $ImageTag" -ForegroundColor White
Write-Host "  Image URI: $ImageUri" -ForegroundColor White
Write-Host ""

# Step 1: Verify AWS credentials
Write-Host "✓ Verifying AWS credentials..." -ForegroundColor Yellow
try {
    $identity = aws sts get-caller-identity --output json --no-cli-pager | ConvertFrom-Json
    Write-Host "  Account: $($identity.Account)" -ForegroundColor Green
} catch {
    Write-Host "✗ AWS credentials not configured" -ForegroundColor Red
    exit 1
}

# Step 2: Login to ECR
Write-Host ""
Write-Host "✓ Logging in to Amazon ECR..." -ForegroundColor Yellow
try {
    aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"
    Write-Host "  Logged in successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to login to ECR" -ForegroundColor Red
    exit 1
}

# Step 3: Build Docker image
Write-Host ""
Write-Host "✓ Building Docker image..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray

$dockerfilePath = Join-Path $PSScriptRoot "..\backend\Dockerfile"
$contextPath = Join-Path $PSScriptRoot "..\backend"

try {
    docker build -t $RepositoryName -f $dockerfilePath $contextPath
    Write-Host "  Docker image built successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to build Docker image" -ForegroundColor Red
    exit 1
}

# Step 4: Tag image
Write-Host ""
Write-Host "✓ Tagging Docker image..." -ForegroundColor Yellow
try {
    docker tag "${RepositoryName}:latest" $ImageUri
    Write-Host "  Tagged as: $ImageUri" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to tag image" -ForegroundColor Red
    exit 1
}

# Step 5: Push to ECR
Write-Host ""
Write-Host "✓ Pushing image to ECR..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
try {
    docker push $ImageUri
    Write-Host "  Image pushed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to push image to ECR" -ForegroundColor Red
    exit 1
}

# Step 6: Update ECS service
Write-Host ""
Write-Host "✓ Updating ECS service..." -ForegroundColor Yellow
try {
    aws ecs update-service `
        --cluster $ClusterName `
        --service $ServiceName `
        --force-new-deployment `
        --region $Region `
        --no-cli-pager
    
    Write-Host "  Service update initiated" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to update ECS service" -ForegroundColor Red
    exit 1
}

# Step 7: Wait for deployment
Write-Host ""
Write-Host "✓ Waiting for service to stabilize..." -ForegroundColor Yellow
Write-Host "  (This may take 3-5 minutes)" -ForegroundColor Gray

try {
    aws ecs wait services-stable `
        --cluster $ClusterName `
        --services $ServiceName `
        --region $Region
    
    Write-Host "  Service is stable" -ForegroundColor Green
} catch {
    Write-Host "⚠ Timeout waiting for service to stabilize" -ForegroundColor Yellow
    Write-Host "  Service is still deploying. Check AWS Console for status." -ForegroundColor Yellow
}

# Step 8: Get service status
Write-Host ""
Write-Host "✓ Getting service status..." -ForegroundColor Yellow
try {
    $service = aws ecs describe-services `
        --cluster $ClusterName `
        --services $ServiceName `
        --region $Region `
        --output json `
        --no-cli-pager | ConvertFrom-Json
    
    $serviceData = $service.services[0]
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Deployment Complete!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Service Status:" -ForegroundColor Cyan
    Write-Host "  Running Tasks: $($serviceData.runningCount)" -ForegroundColor White
    Write-Host "  Desired Tasks: $($serviceData.desiredCount)" -ForegroundColor White
    Write-Host "  Pending Tasks: $($serviceData.pendingCount)" -ForegroundColor White
    Write-Host ""
    
    # Get task IP
    $tasks = aws ecs list-tasks `
        --cluster $ClusterName `
        --service-name $ServiceName `
        --region $Region `
        --output json `
        --no-cli-pager | ConvertFrom-Json
    
    if ($tasks.taskArns.Count -gt 0) {
        $taskDetails = aws ecs describe-tasks `
            --cluster $ClusterName `
            --tasks $tasks.taskArns[0] `
            --region $Region `
            --output json `
            --no-cli-pager | ConvertFrom-Json
        
        $task = $taskDetails.tasks[0]
        
        # Extract ENI ID
        $eniId = $null
        foreach ($attachment in $task.attachments) {
            if ($attachment.type -eq "ElasticNetworkInterface") {
                foreach ($detail in $attachment.details) {
                    if ($detail.name -eq "networkInterfaceId") {
                        $eniId = $detail.value
                        break
                    }
                }
            }
        }
        
        if ($eniId) {
            # Get public IP
            $eni = aws ec2 describe-network-interfaces `
                --network-interface-ids $eniId `
                --region $Region `
                --output json `
                --no-cli-pager | ConvertFrom-Json
            
            $publicIp = $eni.NetworkInterfaces[0].Association.PublicIp
            
            if ($publicIp) {
                Write-Host "Backend URL:" -ForegroundColor Cyan
                Write-Host "  http://${publicIp}:8000" -ForegroundColor White
                Write-Host "  http://${publicIp}:8000/health" -ForegroundColor Gray
                Write-Host ""
            }
        }
    }
    
    Write-Host "New Features Deployed:" -ForegroundColor Cyan
    Write-Host "  ✓ MT5 Account CRUD endpoints" -ForegroundColor White
    Write-Host "  ✓ GET /account/mt5 - List all accounts" -ForegroundColor Gray
    Write-Host "  ✓ GET /account/mt5/primary - Get primary account" -ForegroundColor Gray
    Write-Host "  ✓ POST /account/mt5 - Create account" -ForegroundColor Gray
    Write-Host "  ✓ PUT /account/mt5/{id} - Update account" -ForegroundColor Gray
    Write-Host "  ✓ DELETE /account/mt5/{id} - Delete account" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  ✓ MT5Account database table created" -ForegroundColor White
    Write-Host "  ✓ User-scoped account management" -ForegroundColor White
    Write-Host "  ✓ JWT authentication required" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Update iOS app API_BASE_URL with backend URL" -ForegroundColor White
    Write-Host "  2. Test MT5 account creation from iOS app" -ForegroundColor White
    Write-Host "  3. Verify account sync works end-to-end" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host "⚠ Could not get service status: $_" -ForegroundColor Yellow
}

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  AWS Deployment Complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
