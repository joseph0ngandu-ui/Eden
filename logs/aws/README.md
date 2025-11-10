# AWS Auto-Resilience Logs

This directory is reserved for local AWS auto-resilience activity logs.

## Log Files

All auto-resilience events are primarily logged to CloudWatch, but can be archived here for offline analysis.

### CloudWatch Log Groups

The following log groups contain auto-resilience activity:

- `/aws/lambda/prod-eden-ip-updater` - IP update Lambda logs
- `/aws/lambda/prod-eden-health-monitor` - Health monitor Lambda logs
- `/eden/prod/autoresiliency` - General auto-resilience logs

### Downloading Logs

To download logs from CloudWatch:

```powershell
# Download IP updater logs
aws logs tail /aws/lambda/prod-eden-ip-updater `
    --since 24h `
    --region us-east-1 `
    > ip-updater-$(Get-Date -Format 'yyyy-MM-dd').log

# Download health monitor logs
aws logs tail /aws/lambda/prod-eden-health-monitor `
    --since 24h `
    --region us-east-1 `
    > health-monitor-$(Get-Date -Format 'yyyy-MM-dd').log
```

## Log Retention

- CloudWatch logs are retained for 7-30 days depending on log group
- Local logs should be archived or deleted after analysis
- Do not commit sensitive logs to version control

## .gitignore

This directory's contents (except this README) are ignored by git.
