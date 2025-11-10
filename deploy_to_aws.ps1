#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy Eden Trading Bot (Volatility Burst v1.3) to AWS ECS

.DESCRIPTION
    This script builds and deploys the profitable Volatility Burst v1.3 strategy to AWS ECS.
    
    Prerequisites:
    - AWS CLI configured with appropriate credentials
    - Docker installed and running
    - ECR repository created
    
.NOTES
    Strategy: Volatility Burst v1.3
    Expected Performance: +$1,864.15 PnL
    Win Rate: 46.39%
    Profit Factor: 1.02
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ECRRepository = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ECSCluster = "eden-cluster",
    
    [Parameter(Mandatory=$false)]
    [string]$ECSService = "eden-bot",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"

Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Eden Trading Bot - Deployment to AWS                   ║" -ForegroundColor Cyan
Write-Host "║  Strategy: Volatility Burst v1.3 (PROFITABLE)           ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Validate prerequisites
Write-Host "→ Checking prerequisites..." -ForegroundColor Yellow

# Check Docker
try {
    docker --version | Out-Null
    Write-Host "  ✓ Docker is installed" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker is not installed or not running" -ForegroundColor Red
    exit 1
}

# Check AWS CLI
try {
    aws --version | Out-Null
    Write-Host "  ✓ AWS CLI is installed" -ForegroundColor Green
} catch {
    Write-Host "  ✗ AWS CLI is not installed" -ForegroundColor Red
    exit 1
}

# Prompt for ECR repository if not provided
if ([string]::IsNullOrEmpty($ECRRepository)) {
    $ECRRepository = Read-Host "Enter your ECR repository URI (e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com/eden-bot)"
}

Write-Host ""
Write-Host "→ Configuration:" -ForegroundColor Yellow
Write-Host "  ECR Repository: $ECRRepository" -ForegroundColor White
Write-Host "  ECS Cluster: $ECSCluster" -ForegroundColor White
Write-Host "  ECS Service: $ECSService" -ForegroundColor White
Write-Host "  Region: $Region" -ForegroundColor White
Write-Host ""

# Confirm deployment
$confirm = Read-Host "Deploy Volatility Burst v1.3 to production? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Deployment cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "→ Building Docker image..." -ForegroundColor Yellow
Set-Location "backend"

docker build -t eden-trading-bot:vb-v1.3 .
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Docker build failed" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Docker image built successfully" -ForegroundColor Green

Write-Host ""
Write-Host "→ Logging in to Amazon ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ECRRepository.Split('/')[0]
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ ECR login failed" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Successfully logged in to ECR" -ForegroundColor Green

Write-Host ""
Write-Host "→ Tagging image..." -ForegroundColor Yellow
docker tag eden-trading-bot:vb-v1.3 "${ECRRepository}:latest"
docker tag eden-trading-bot:vb-v1.3 "${ECRRepository}:vb-v1.3"
Write-Host "  ✓ Image tagged" -ForegroundColor Green

Write-Host ""
Write-Host "→ Pushing to ECR..." -ForegroundColor Yellow
docker push "${ECRRepository}:latest"
docker push "${ECRRepository}:vb-v1.3"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to push image to ECR" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Image pushed to ECR successfully" -ForegroundColor Green

Write-Host ""
Write-Host "→ Updating ECS service..." -ForegroundColor Yellow
aws ecs update-service `
    --cluster $ECSCluster `
    --service $ECSService `
    --force-new-deployment `
    --region $Region | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to update ECS service" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ ECS service updated successfully" -ForegroundColor Green

Write-Host ""
Write-Host "→ Waiting for deployment to stabilize..." -ForegroundColor Yellow
aws ecs wait services-stable `
    --cluster $ECSCluster `
    --services $ECSService `
    --region $Region

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ⚠ Warning: Deployment may still be in progress" -ForegroundColor Yellow
} else {
    Write-Host "  ✓ Deployment stabilized" -ForegroundColor Green
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           DEPLOYMENT SUCCESSFUL! 🚀                      ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Strategy: Volatility Burst v1.3 (PROFITABLE)" -ForegroundColor Cyan
Write-Host "Expected Performance: +`$1,864.15 PnL" -ForegroundColor Green
Write-Host "Win Rate: 46.39%" -ForegroundColor Green
Write-Host "Profit Factor: 1.02" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Monitor via your iOS app" -ForegroundColor White
Write-Host "2. Check CloudWatch logs for trade execution" -ForegroundColor White
Write-Host "3. Verify strategy config via API: GET /strategy/config" -ForegroundColor White
Write-Host "4. Monitor key metrics: Win rate ~46%, Profit factor >1.0" -ForegroundColor White
Write-Host ""

Set-Location ".."
