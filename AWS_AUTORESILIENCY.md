# Eden AWS Auto-Resilient Infrastructure

## Overview

This document describes the fully automated, self-healing production infrastructure for Eden on AWS. The system provides:

- ✅ **Automatic API Gateway IP updates** when ECS tasks restart or scale
- ✅ **Health monitoring** with automatic ECS task restarts on failures
- ✅ **CloudWatch logging** of all auto-resilience activities
- ✅ **Optional SNS notifications** for all automated actions
- ✅ **Scheduled health checks** every 5 minutes
- ✅ **Zero-downtime** deployments with automatic recovery

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EDEN AUTO-RESILIENT SYSTEM                │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│   ECS Service    │────────▶│  EventBridge     │
│  Task Changes    │  Events │  ECS State Rule  │
└──────────────────┘         └────────┬─────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │  IP Update Lambda   │
                            │  • Fetch new IP     │
                            │  • Update API GW    │
                            │  • Log to CloudWatch│
                            └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │   API Gateway       │
                            │  (Auto-Updated URI) │
                            └─────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  API Gateway     │────────▶│  CloudWatch      │
│  5XX Errors      │ Metrics │  Alarm           │
└──────────────────┘         └────────┬─────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │  EventBridge        │
                            │  Alarm State Rule   │
                            └─────────┬───────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │ Health Monitor      │
                            │ Lambda              │
┌──────────────────┐        │ • Check service     │
│  Scheduled       │───────▶│ • Stop unhealthy    │
│  Every 5 min     │        │ • Force redeploy    │
└──────────────────┘        │ • Send SNS alert    │
                            └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │   SNS Topic         │
                            │  (Email Alerts)     │
                            └─────────────────────┘
```

## Components

### 1. IP Update Lambda Function

**Purpose**: Automatically updates API Gateway integration URI when ECS task IP changes.

**Triggers**:
- ECS Task State Change events (RUNNING/STOPPED)

**Actions**:
1. Detects ECS task state changes
2. Fetches new public IP from ECS task ENI
3. Updates API Gateway integration URI
4. Logs all actions to CloudWatch

**Environment Variables**:
- `API_GATEWAY_ID`: API Gateway ID
- `INTEGRATION_ID`: API Gateway integration ID
- `ECS_CLUSTER`: ECS cluster name
- `ECS_SERVICE`: ECS service name
- `BACKEND_PORT`: Backend port (default: 8000)

**Timeout**: 60 seconds  
**Memory**: 256 MB

### 2. Health Monitor Lambda Function

**Purpose**: Monitors ECS service health and automatically restarts unhealthy tasks.

**Triggers**:
- CloudWatch Alarm state changes (API Gateway 5XX errors)
- Scheduled: Every 5 minutes
- Manual invocation for testing

**Actions**:
1. Checks ECS service health (running vs desired task count)
2. Identifies unhealthy tasks
3. Stops unhealthy tasks (ECS auto-restarts them)
4. Forces service redeployment if no specific unhealthy tasks found
5. Sends SNS notifications on all actions
6. Logs everything to CloudWatch

**Environment Variables**:
- `ECS_CLUSTER`: ECS cluster name
- `ECS_SERVICE`: ECS service name
- `API_GATEWAY_URL`: API Gateway URL for monitoring
- `SNS_TOPIC_ARN`: SNS topic for notifications (optional)

**Timeout**: 120 seconds  
**Memory**: 256 MB

### 3. EventBridge Rules

#### ECS Task State Change Rule
- **Name**: `{env}-eden-ecs-task-state-change`
- **Event Pattern**: ECS Task State Change (RUNNING/STOPPED)
- **Target**: IP Update Lambda

#### Health Alarm Trigger Rule
- **Name**: `{env}-eden-health-alarm-trigger`
- **Event Pattern**: CloudWatch Alarm State Change (ALARM state)
- **Target**: Health Monitor Lambda

#### Scheduled Health Check Rule
- **Name**: `{env}-eden-scheduled-health-check`
- **Schedule**: Rate(5 minutes)
- **Target**: Health Monitor Lambda

### 4. CloudWatch Alarm

- **Name**: `{env}-eden-api-health-alarm`
- **Metric**: API Gateway 5XXError
- **Threshold**: ≥ 1 error
- **Period**: 60 seconds
- **Evaluation Periods**: 2 consecutive failures
- **Action**: Triggers Health Monitor Lambda

### 5. SNS Topic (Optional)

- **Name**: `{env}-eden-autoresiliency-notifications`
- **Purpose**: Email notifications for auto-healing actions
- **Subscriptions**: Configured email address

## Deployment

### Prerequisites

1. AWS CLI configured with appropriate credentials
2. Existing ECS cluster and service
3. Existing API Gateway with HTTP integration
4. PowerShell 7.0+ (for Windows)

### Required Information

Before deployment, gather:

- **API Gateway ID**: From AWS Console → API Gateway → Your API
- **API Gateway Integration ID**: From Integration settings
- **API Gateway URL**: Full URL (e.g., `https://abc123.execute-api.us-east-1.amazonaws.com/prod`)
- **Notification Email** (optional): Email for alerts

### Deployment Steps

#### 1. Deploy CloudFormation Stack

```powershell
.\scripts\deploy-autoresiliency.ps1 `
    -APIGatewayId "your-api-gateway-id" `
    -APIGatewayIntegrationId "your-integration-id" `
    -APIGatewayURL "https://your-api-gateway-url.com/prod" `
    -NotificationEmail "your-email@example.com" `
    -Environment "prod" `
    -Region "us-east-1"
```

**Parameters**:
- `APIGatewayId`: (Required) Your API Gateway ID
- `APIGatewayIntegrationId`: (Required) Your Integration ID
- `APIGatewayURL`: (Required) Your API Gateway URL
- `NotificationEmail`: (Optional) Email for SNS notifications
- `Environment`: (Optional) Environment name (default: prod)
- `Region`: (Optional) AWS region (default: us-east-1)

#### 2. Confirm SNS Email Subscription

If you provided a notification email:
1. Check your email inbox
2. Look for "AWS Notification - Subscription Confirmation"
3. Click the confirmation link

#### 3. Verify Deployment

```powershell
.\scripts\test-autoresiliency.ps1 -Environment "prod" -Region "us-east-1"
```

The test script will verify:
- Lambda functions exist and are configured
- EventBridge rules are enabled
- CloudWatch alarm is active
- ECS service is healthy
- CloudWatch logs are being written

## Testing

### Test 1: Auto-IP Update

Test that API Gateway automatically updates when ECS task restarts:

```powershell
# Get current task ID
aws ecs list-tasks `
    --cluster prod-eden-cluster `
    --service prod-eden-service `
    --region us-east-1

# Stop the task (it will auto-restart)
aws ecs stop-task `
    --cluster prod-eden-cluster `
    --task <task-arn> `
    --reason "Testing auto-IP update" `
    --region us-east-1

# Check Lambda logs
aws logs tail /aws/lambda/prod-eden-ip-updater `
    --follow `
    --region us-east-1
```

**Expected Result**:
- Task stops and restarts automatically
- IP Update Lambda is triggered by EventBridge
- API Gateway integration URI is updated with new IP
- CloudWatch logs show successful update

### Test 2: Health Monitoring

Test that health monitor detects and fixes issues:

```powershell
# Manually invoke health monitor
aws lambda invoke `
    --function-name prod-eden-health-monitor `
    --region us-east-1 `
    --payload '{"source":"manual-test"}' `
    response.json

# View response
cat response.json

# Check logs
aws logs tail /aws/lambda/prod-eden-health-monitor `
    --follow `
    --region us-east-1
```

**Expected Result**:
- Lambda checks service health
- Reports running/desired task count
- If healthy: No action, logs success
- If unhealthy: Stops tasks or forces redeployment
- SNS notification sent (if configured)

### Test 3: Simulated Failure

Simulate API failure by scaling service to 0:

```powershell
# Scale down to 0 tasks (temporary)
aws ecs update-service `
    --cluster prod-eden-cluster `
    --service prod-eden-service `
    --desired-count 0 `
    --region us-east-1

# Wait for health check alarm (1-2 minutes)
# Health monitor will detect and auto-scale back

# Restore manually if needed
aws ecs update-service `
    --cluster prod-eden-cluster `
    --service prod-eden-service `
    --desired-count 1 `
    --region us-east-1
```

**Expected Result**:
- CloudWatch alarm triggers after 2 consecutive periods
- Health Monitor Lambda is invoked
- Lambda detects task count mismatch
- Lambda forces service redeployment
- SNS notification sent

## Monitoring

### CloudWatch Logs

All auto-resilience activities are logged:

**IP Update Lambda Logs**:
```
/aws/lambda/{env}-eden-ip-updater
```

**Health Monitor Lambda Logs**:
```
/aws/lambda/{env}-eden-health-monitor
```

**Auto-Resilience Log Group**:
```
/eden/{env}/autoresiliency
```

### CloudWatch Metrics

Monitor Lambda invocations:
- **Namespace**: AWS/Lambda
- **Metrics**: Invocations, Errors, Duration

Monitor API Gateway health:
- **Namespace**: AWS/ApiGateway
- **Metrics**: 5XXError, 4XXError, Latency

### CloudWatch Alarms

Check alarm status:

```powershell
aws cloudwatch describe-alarms `
    --alarm-names "prod-eden-api-health-alarm" `
    --region us-east-1
```

### EventBridge Monitoring

View rule invocations in AWS Console:
- EventBridge → Rules → Select rule → Metrics

## SNS Notifications

If configured, you'll receive email notifications for:

### Auto-Healing Triggered
```
Subject: 🚨 Eden Service Auto-Healing Triggered

Eden Health Monitor Alert

Service: prod-eden-service
Cluster: prod-eden-cluster
Status: UNHEALTHY

Health Check Results:
- Running Tasks: 0/1
- Reason: Task count mismatch: 0/1

Actions Taken:
- Forced service redeployment

Timestamp: 2025-11-10T17:30:00Z
API Gateway: https://your-api-gateway-url.com/prod
```

### Health Monitor Error
```
Subject: 🚨 Eden Health Monitor Error

Error in health monitor Lambda:

[Error details]

Timestamp: 2025-11-10T17:30:00Z
```

## Costs

Estimated monthly costs (us-east-1 pricing):

| Service | Usage | Cost |
|---------|-------|------|
| Lambda (IP Update) | ~100 invocations/month | $0.00 (free tier) |
| Lambda (Health Monitor) | ~8,640 invocations/month (5 min) | $0.00 (free tier) |
| CloudWatch Logs | ~1 GB/month | $0.50 |
| CloudWatch Alarms | 1 alarm | $0.10 |
| EventBridge | ~10,000 events/month | $0.00 (free tier) |
| SNS | ~100 notifications/month | $0.00 (free tier) |
| **Total** | | **~$0.60/month** |

## Troubleshooting

### Issue: IP Update Lambda Not Triggering

**Symptoms**: ECS task restarts but API Gateway URI not updated

**Check**:
1. EventBridge rule is enabled:
   ```powershell
   aws events describe-rule --name prod-eden-ecs-task-state-change --region us-east-1
   ```
2. Lambda has correct permissions (IAM role)
3. Lambda environment variables are set correctly

**Fix**:
- Re-deploy CloudFormation stack
- Check Lambda logs for errors

### Issue: Health Monitor Not Auto-Healing

**Symptoms**: Service unhealthy but no automatic restart

**Check**:
1. CloudWatch alarm is active and firing
2. EventBridge rule for alarm state change is enabled
3. Lambda has ECS permissions (StopTask, UpdateService)

**Fix**:
- Manually invoke Lambda to test:
  ```powershell
  aws lambda invoke --function-name prod-eden-health-monitor response.json
  ```
- Check IAM permissions

### Issue: SNS Notifications Not Received

**Symptoms**: Auto-healing works but no email notifications

**Check**:
1. SNS subscription is confirmed (check email)
2. Lambda environment variable `SNS_TOPIC_ARN` is set
3. Lambda has SNS publish permissions

**Fix**:
- Confirm SNS subscription via email link
- Re-deploy stack with notification email

### Issue: High Lambda Costs

**Symptoms**: Unexpected AWS charges

**Check**:
- Lambda invocation count in CloudWatch metrics
- Scheduled health check frequency (default: 5 minutes)

**Fix**:
- Adjust scheduled rule frequency:
  ```powershell
  aws events put-rule `
      --name prod-eden-scheduled-health-check `
      --schedule-expression "rate(10 minutes)" `
      --region us-east-1
  ```

## Maintenance

### Updating Lambda Functions

To update Lambda code:

1. Update code in `aws-autoresiliency.yml` (inline code section)
2. Re-deploy CloudFormation stack:
   ```powershell
   .\scripts\deploy-autoresiliency.ps1 [parameters]
   ```

### Changing Health Check Frequency

Edit CloudFormation template:
```yaml
ScheduledHealthCheckRule:
  Type: AWS::Events::Rule
  Properties:
    ScheduleExpression: rate(10 minutes)  # Change from 5 to 10
```

### Disabling Auto-Healing Temporarily

Disable EventBridge rules:

```powershell
# Disable health alarm trigger
aws events disable-rule `
    --name prod-eden-health-alarm-trigger `
    --region us-east-1

# Disable scheduled health check
aws events disable-rule `
    --name prod-eden-scheduled-health-check `
    --region us-east-1
```

Re-enable when ready:

```powershell
aws events enable-rule --name prod-eden-health-alarm-trigger --region us-east-1
aws events enable-rule --name prod-eden-scheduled-health-check --region us-east-1
```

## Cleanup

To remove all auto-resilience infrastructure:

```powershell
aws cloudformation delete-stack `
    --stack-name eden-autoresiliency `
    --region us-east-1

# Wait for deletion
aws cloudformation wait stack-delete-complete `
    --stack-name eden-autoresiliency `
    --region us-east-1
```

**Note**: This will NOT delete your ECS service, API Gateway, or other core infrastructure.

## Security Best Practices

1. **IAM Least Privilege**: Lambda functions have minimum required permissions
2. **CloudWatch Logging**: All actions are logged for audit trails
3. **Secrets Management**: No hardcoded credentials or secrets
4. **Network Security**: Uses existing VPC and security group configurations
5. **Resource Tagging**: All resources tagged with environment and purpose

## Support

For issues or questions:

1. Check CloudWatch logs first
2. Review this documentation
3. Run test script: `.\scripts\test-autoresiliency.ps1`
4. Contact DevOps team with logs and error messages

## Changelog

### Version 1.0 (2025-11-10)
- Initial release
- Auto IP update on ECS task changes
- Health monitoring with CloudWatch alarms
- Automatic task restart on failures
- SNS notifications
- Scheduled health checks every 5 minutes
- Comprehensive logging

## Related Documentation

- [AWS_SETUP.md](./AWS_SETUP.md) - AWS infrastructure setup
- [aws-infrastructure.yml](./aws-infrastructure.yml) - Main infrastructure template
- [aws-infrastructure-simple.yml](./aws-infrastructure-simple.yml) - Simplified infrastructure

## License

This infrastructure code is part of the Eden Trading Bot project.
