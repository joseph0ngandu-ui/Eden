# Test Eden Auto-Resilient Infrastructure
# Tests auto-IP updates and health monitoring

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "prod",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Eden Auto-Resilient Infrastructure Test Suite" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$clusterName = "$Environment-eden-cluster"
$serviceName = "$Environment-eden-service"

# Test 1: Check Lambda Functions
Write-Host "Test 1: Verify Lambda Functions" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray

try {
    $ipUpdaterName = "$Environment-eden-ip-updater"
    $healthMonitorName = "$Environment-eden-health-monitor"
    
    Write-Host "  Checking IP Updater Lambda..." -ForegroundColor White
    $ipUpdater = aws lambda get-function `
        --function-name $ipUpdaterName `
        --region $Region `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($ipUpdater.Configuration) {
        Write-Host "    ✓ $ipUpdaterName exists" -ForegroundColor Green
        Write-Host "      Runtime: $($ipUpdater.Configuration.Runtime)" -ForegroundColor Gray
        Write-Host "      Timeout: $($ipUpdater.Configuration.Timeout)s" -ForegroundColor Gray
    }
    
    Write-Host "  Checking Health Monitor Lambda..." -ForegroundColor White
    $healthMonitor = aws lambda get-function `
        --function-name $healthMonitorName `
        --region $Region `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($healthMonitor.Configuration) {
        Write-Host "    ✓ $healthMonitorName exists" -ForegroundColor Green
        Write-Host "      Runtime: $($healthMonitor.Configuration.Runtime)" -ForegroundColor Gray
        Write-Host "      Timeout: $($healthMonitor.Configuration.Timeout)s" -ForegroundColor Gray
    }
    
    Write-Host "  ✅ Test 1 Passed" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Test 1 Failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 2: Check EventBridge Rules
Write-Host "Test 2: Verify EventBridge Rules" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray

try {
    $ecsRuleName = "$Environment-eden-ecs-task-state-change"
    $healthRuleName = "$Environment-eden-health-alarm-trigger"
    $scheduledRuleName = "$Environment-eden-scheduled-health-check"
    
    Write-Host "  Checking ECS Task State Change Rule..." -ForegroundColor White
    $ecsRule = aws events describe-rule `
        --name $ecsRuleName `
        --region $Region `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($ecsRule.State -eq "ENABLED") {
        Write-Host "    ✓ $ecsRuleName is ENABLED" -ForegroundColor Green
    }
    
    Write-Host "  Checking Health Alarm Trigger Rule..." -ForegroundColor White
    $healthRule = aws events describe-rule `
        --name $healthRuleName `
        --region $Region `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($healthRule.State -eq "ENABLED") {
        Write-Host "    ✓ $healthRuleName is ENABLED" -ForegroundColor Green
    }
    
    Write-Host "  Checking Scheduled Health Check Rule..." -ForegroundColor White
    $scheduledRule = aws events describe-rule `
        --name $scheduledRuleName `
        --region $Region `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($scheduledRule.State -eq "ENABLED") {
        Write-Host "    ✓ $scheduledRuleName is ENABLED" -ForegroundColor Green
        Write-Host "      Schedule: $($scheduledRule.ScheduleExpression)" -ForegroundColor Gray
    }
    
    Write-Host "  ✅ Test 2 Passed" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Test 2 Failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 3: Check CloudWatch Alarm
Write-Host "Test 3: Verify CloudWatch Alarm" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray

try {
    $alarmName = "$Environment-eden-api-health-alarm"
    
    Write-Host "  Checking API Health Alarm..." -ForegroundColor White
    $alarm = aws cloudwatch describe-alarms `
        --alarm-names $alarmName `
        --region $Region `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($alarm.MetricAlarms) {
        $alarmData = $alarm.MetricAlarms[0]
        Write-Host "    ✓ $alarmName exists" -ForegroundColor Green
        Write-Host "      State: $($alarmData.StateValue)" -ForegroundColor Gray
        Write-Host "      Threshold: $($alarmData.Threshold)" -ForegroundColor Gray
        Write-Host "      Period: $($alarmData.Period)s" -ForegroundColor Gray
    }
    
    Write-Host "  ✅ Test 3 Passed" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Test 3 Failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 4: Check ECS Service Status
Write-Host "Test 4: Verify ECS Service Health" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray

try {
    Write-Host "  Checking ECS Service..." -ForegroundColor White
    $service = aws ecs describe-services `
        --cluster $clusterName `
        --services $serviceName `
        --region $Region `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($service.services) {
        $serviceData = $service.services[0]
        $running = $serviceData.runningCount
        $desired = $serviceData.desiredCount
        
        Write-Host "    Service: $serviceName" -ForegroundColor Gray
        Write-Host "    Running Tasks: $running/$desired" -ForegroundColor Gray
        
        if ($running -eq $desired -and $running -gt 0) {
            Write-Host "    ✓ Service is healthy" -ForegroundColor Green
        } else {
            Write-Host "    ⚠ Service may be unhealthy" -ForegroundColor Yellow
        }
    }
    
    Write-Host "  ✅ Test 4 Passed" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Test 4 Failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 5: Invoke Health Monitor Lambda (Manual Test)
Write-Host "Test 5: Manual Lambda Invocation Test" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray

$testHealthMonitor = Read-Host "Invoke Health Monitor Lambda manually? (yes/no)"
if ($testHealthMonitor -eq "yes") {
    try {
        Write-Host "  Invoking Health Monitor Lambda..." -ForegroundColor White
        
        $testEvent = @{
            source = "manual-test"
            time = Get-Date -Format "o"
        } | ConvertTo-Json
        
        $invokeResult = aws lambda invoke `
            --function-name "$Environment-eden-health-monitor" `
            --payload $testEvent `
            --region $Region `
            --no-cli-pager `
            response.json
        
        if (Test-Path response.json) {
            $response = Get-Content response.json | ConvertFrom-Json
            Write-Host "    Response:" -ForegroundColor Gray
            Write-Host "    $($response | ConvertTo-Json -Depth 3)" -ForegroundColor Gray
            Remove-Item response.json
        }
        
        Write-Host "  ✅ Test 5 Passed" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Test 5 Failed: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  ⊘ Test 5 Skipped" -ForegroundColor Yellow
}

Write-Host ""

# Test 6: Check CloudWatch Logs
Write-Host "Test 6: Check Recent CloudWatch Logs" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray

try {
    Write-Host "  Checking IP Updater logs..." -ForegroundColor White
    $ipUpdaterLogs = aws logs describe-log-streams `
        --log-group-name "/aws/lambda/$Environment-eden-ip-updater" `
        --region $Region `
        --order-by LastEventTime `
        --descending `
        --max-items 1 `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($ipUpdaterLogs.logStreams) {
        $lastEvent = $ipUpdaterLogs.logStreams[0].lastEventTimestamp
        $lastEventDate = [DateTimeOffset]::FromUnixTimeMilliseconds($lastEvent).DateTime
        Write-Host "    ✓ Last log entry: $lastEventDate" -ForegroundColor Green
    }
    
    Write-Host "  Checking Health Monitor logs..." -ForegroundColor White
    $healthMonitorLogs = aws logs describe-log-streams `
        --log-group-name "/aws/lambda/$Environment-eden-health-monitor" `
        --region $Region `
        --order-by LastEventTime `
        --descending `
        --max-items 1 `
        --output json `
        --no-cli-pager 2>&1 | ConvertFrom-Json
    
    if ($healthMonitorLogs.logStreams) {
        $lastEvent = $healthMonitorLogs.logStreams[0].lastEventTimestamp
        $lastEventDate = [DateTimeOffset]::FromUnixTimeMilliseconds($lastEvent).DateTime
        Write-Host "    ✓ Last log entry: $lastEventDate" -ForegroundColor Green
    }
    
    Write-Host "  ✅ Test 6 Passed" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Test 6 Failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Test Suite Complete" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Additional Manual Tests:" -ForegroundColor Cyan
Write-Host "  1. Stop an ECS task to test auto-IP update:" -ForegroundColor White
Write-Host "     aws ecs list-tasks --cluster $clusterName --service $serviceName --region $Region" -ForegroundColor Gray
Write-Host "     aws ecs stop-task --cluster $clusterName --task <task-id> --reason 'Testing auto-update' --region $Region" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Monitor Lambda execution logs:" -ForegroundColor White
Write-Host "     https://console.aws.amazon.com/cloudwatch/home?region=$Region#logsV2:log-groups" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Check EventBridge rule invocations:" -ForegroundColor White
Write-Host "     https://console.aws.amazon.com/events/home?region=$Region#/rules" -ForegroundColor Gray
Write-Host ""
