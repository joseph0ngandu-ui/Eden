# Eden Auto-Resilient Infrastructure - Quick Start

## 🚀 Deploy in 5 Minutes

This guide will help you deploy the full auto-resilient infrastructure for Eden on AWS.

## Prerequisites

✅ AWS CLI configured  
✅ Existing ECS cluster and service  
✅ Existing API Gateway  
✅ PowerShell 7.0+

## Step 1: Gather Information

You need these values from AWS Console:

### API Gateway Information

1. Go to **API Gateway Console** → Your API
2. Copy **API ID** (e.g., `mstqazdmha`)
3. Go to **Integrations** tab
4. Copy **Integration ID** (looks like `abc123xyz`)
5. Copy the full **API URL** (e.g., `https://mstqazdmha.execute-api.us-east-1.amazonaws.com/prod`)

### ECS Information (Auto-detected by script)

- ECS Cluster: `prod-eden-cluster`
- ECS Service: `prod-eden-service`

## Step 2: Deploy

Run the deployment script:

```powershell
cd "C:\Users\Sal\OneDrive - ZCAS University\Eden"

.\scripts\deploy-autoresiliency.ps1 `
    -APIGatewayId "YOUR_API_GATEWAY_ID" `
    -APIGatewayIntegrationId "YOUR_INTEGRATION_ID" `
    -APIGatewayURL "https://YOUR_API_GATEWAY_URL.com/prod" `
    -NotificationEmail "your-email@example.com"
```

**Example:**
```powershell
.\scripts\deploy-autoresiliency.ps1 `
    -APIGatewayId "mstqazdmha" `
    -APIGatewayIntegrationId "abc123xyz" `
    -APIGatewayURL "https://mstqazdmha.execute-api.us-east-1.amazonaws.com/prod" `
    -NotificationEmail "sal@example.com"
```

The script will:
- ✅ Verify AWS credentials
- ✅ Validate ECS cluster/service
- ✅ Deploy CloudFormation stack (3-5 minutes)
- ✅ Create all Lambda functions
- ✅ Set up EventBridge rules
- ✅ Configure CloudWatch alarms
- ✅ Set up SNS notifications

## Step 3: Confirm Email (Optional)

If you provided an email:
1. Check your inbox for "AWS Notification - Subscription Confirmation"
2. Click the confirmation link

## Step 4: Test the System

Run the test suite:

```powershell
.\scripts\test-autoresiliency.ps1
```

This verifies:
- ✅ Lambda functions deployed
- ✅ EventBridge rules enabled
- ✅ CloudWatch alarms active
- ✅ ECS service healthy

## Step 5: Test Auto-IP Update (Optional)

Manually stop an ECS task to test:

```powershell
# List tasks
aws ecs list-tasks `
    --cluster prod-eden-cluster `
    --service prod-eden-service `
    --region us-east-1

# Stop a task (it will auto-restart)
aws ecs stop-task `
    --cluster prod-eden-cluster `
    --task <TASK_ARN_FROM_ABOVE> `
    --reason "Testing auto-update" `
    --region us-east-1

# Watch Lambda logs
aws logs tail /aws/lambda/prod-eden-ip-updater --follow --region us-east-1
```

**Expected:** Task restarts, Lambda updates API Gateway with new IP.

## What You Get

### ✅ Automatic IP Updates
- ECS task restarts/scales → EventBridge triggers Lambda → API Gateway updated

### ✅ Health Monitoring
- API Gateway errors → CloudWatch Alarm → Lambda restarts unhealthy tasks
- Scheduled health checks every 5 minutes

### ✅ Notifications
- Email alerts for all auto-healing actions
- CloudWatch logs for audit trail

### ✅ Zero Downtime
- Automatic recovery from failures
- No manual intervention required

## Monitoring

### CloudWatch Logs
```powershell
# IP Update Lambda
aws logs tail /aws/lambda/prod-eden-ip-updater --follow --region us-east-1

# Health Monitor Lambda
aws logs tail /aws/lambda/prod-eden-health-monitor --follow --region us-east-1
```

### CloudWatch Console
[https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups](https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups)

## Costs

**~$0.60/month** (most services in free tier)

- Lambda invocations: Free tier
- CloudWatch Logs: ~$0.50/month
- CloudWatch Alarms: $0.10/month
- EventBridge: Free tier
- SNS: Free tier

## Troubleshooting

### Deployment Fails

**Check AWS credentials:**
```powershell
aws sts get-caller-identity
```

**Check IAM permissions:**
- CloudFormation
- Lambda
- EventBridge
- CloudWatch
- ECS
- SNS

### Lambda Not Triggering

**Check EventBridge rules:**
```powershell
aws events list-rules --region us-east-1
```

**Manually invoke Lambda:**
```powershell
aws lambda invoke `
    --function-name prod-eden-health-monitor `
    --payload '{"source":"test"}' `
    --region us-east-1 `
    response.json
```

### No Email Notifications

**Confirm SNS subscription** - check your email and click confirmation link

## Next Steps

1. **Test thoroughly** - Simulate failures to verify auto-healing
2. **Monitor logs** - Review CloudWatch logs regularly
3. **Adjust frequency** - Change health check interval if needed (default: 5 min)
4. **Push to production** - Your system is now fully resilient!

## Documentation

- **Full Documentation:** [AWS_AUTORESILIENCY.md](./AWS_AUTORESILIENCY.md)
- **AWS Setup:** [AWS_SETUP.md](./AWS_SETUP.md)

## Support

Issues? Run diagnostics:
```powershell
.\scripts\test-autoresiliency.ps1
```

Check logs and review documentation.

---

**🎉 Congratulations!** Your Eden backend is now fully auto-resilient with zero-downtime deployments and automatic recovery.
