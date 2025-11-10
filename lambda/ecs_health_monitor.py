import json
import os
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
ecs_client = boto3.client('ecs')
sns_client = boto3.client('sns')

# Environment variables
ECS_CLUSTER = os.environ['ECS_CLUSTER']
ECS_SERVICE = os.environ['ECS_SERVICE']
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
API_GATEWAY_URL = os.environ.get('API_GATEWAY_URL', '')

def check_ecs_service_health() -> Dict[str, Any]:
    """Check the health status of the ECS service."""
    try:
        logger.info(f"Checking ECS service health: {ECS_SERVICE} in cluster: {ECS_CLUSTER}")
        
        # Describe the service
        response = ecs_client.describe_services(
            cluster=ECS_CLUSTER,
            services=[ECS_SERVICE]
        )
        
        if not response['services']:
            logger.error("Service not found")
            return {
                'healthy': False,
                'reason': 'Service not found',
                'running_tasks': 0,
                'desired_tasks': 0
            }
        
        service = response['services'][0]
        running_count = service['runningCount']
        desired_count = service['desiredCount']
        
        logger.info(f"Service status - Running: {running_count}, Desired: {desired_count}")
        
        # Check if running tasks match desired count
        is_healthy = running_count >= desired_count and running_count > 0
        
        return {
            'healthy': is_healthy,
            'reason': 'Service healthy' if is_healthy else f'Task count mismatch: {running_count}/{desired_count}',
            'running_tasks': running_count,
            'desired_tasks': desired_count,
            'service_arn': service['serviceArn']
        }
        
    except Exception as e:
        logger.error(f"Error checking ECS service health: {str(e)}")
        return {
            'healthy': False,
            'reason': f'Error: {str(e)}',
            'running_tasks': 0,
            'desired_tasks': 0
        }

def get_unhealthy_tasks() -> list:
    """Get list of unhealthy tasks."""
    try:
        # List all tasks in the service
        response = ecs_client.list_tasks(
            cluster=ECS_CLUSTER,
            serviceName=ECS_SERVICE
        )
        
        if not response['taskArns']:
            logger.warning("No tasks found in service")
            return []
        
        # Describe tasks to check health
        task_details = ecs_client.describe_tasks(
            cluster=ECS_CLUSTER,
            tasks=response['taskArns']
        )
        
        unhealthy_tasks = []
        for task in task_details['tasks']:
            last_status = task.get('lastStatus')
            health_status = task.get('healthStatus', 'UNKNOWN')
            
            # Consider task unhealthy if not running or health check failing
            if last_status != 'RUNNING' or health_status == 'UNHEALTHY':
                unhealthy_tasks.append({
                    'taskArn': task['taskArn'],
                    'lastStatus': last_status,
                    'healthStatus': health_status
                })
                logger.info(f"Found unhealthy task: {task['taskArn']} - Status: {last_status}, Health: {health_status}")
        
        return unhealthy_tasks
        
    except Exception as e:
        logger.error(f"Error getting unhealthy tasks: {str(e)}")
        return []

def restart_unhealthy_tasks(unhealthy_tasks: list) -> Dict[str, Any]:
    """Stop unhealthy tasks to trigger automatic restart."""
    stopped_tasks = []
    errors = []
    
    for task in unhealthy_tasks:
        try:
            task_arn = task['taskArn']
            logger.info(f"Stopping unhealthy task: {task_arn}")
            
            response = ecs_client.stop_task(
                cluster=ECS_CLUSTER,
                task=task_arn,
                reason='Auto-healing: Task unhealthy, restarting'
            )
            
            stopped_tasks.append({
                'taskArn': task_arn,
                'stoppedAt': datetime.utcnow().isoformat()
            })
            
            logger.info(f"✅ Successfully stopped task: {task_arn}")
            
            # Small delay between stops
            time.sleep(2)
            
        except Exception as e:
            error_msg = f"Failed to stop task {task['taskArn']}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
    
    return {
        'stopped_tasks': stopped_tasks,
        'stopped_count': len(stopped_tasks),
        'errors': errors
    }

def force_service_update() -> bool:
    """Force a new deployment of the service."""
    try:
        logger.info(f"Forcing service update for {ECS_SERVICE}")
        
        response = ecs_client.update_service(
            cluster=ECS_CLUSTER,
            service=ECS_SERVICE,
            forceNewDeployment=True
        )
        
        logger.info("✅ Service update initiated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update service: {str(e)}")
        return False

def send_sns_notification(subject: str, message: str):
    """Send SNS notification if topic ARN is configured."""
    if not SNS_TOPIC_ARN:
        logger.info("SNS topic not configured, skipping notification")
        return
    
    try:
        logger.info(f"Sending SNS notification: {subject}")
        
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        
        logger.info("✅ SNS notification sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to send SNS notification: {str(e)}")

def lambda_handler(event, context):
    """Main Lambda handler triggered by CloudWatch alarms or EventBridge."""
    try:
        logger.info(f"=== Eden Health Monitor Started ===")
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Determine event source
        event_source = event.get('source', '')
        alarm_name = None
        
        if 'detail-type' in event and 'CloudWatch Alarm' in event['detail-type']:
            # CloudWatch Alarm event
            alarm_name = event.get('detail', {}).get('alarmName', 'Unknown')
            alarm_state = event.get('detail', {}).get('state', {}).get('value', 'Unknown')
            logger.info(f"CloudWatch Alarm triggered: {alarm_name} - State: {alarm_state}")
        
        # Check ECS service health
        health_status = check_ecs_service_health()
        logger.info(f"Service health check: {health_status}")
        
        actions_taken = []
        
        # If service is unhealthy, take action
        if not health_status['healthy']:
            logger.warning(f"⚠️ Service unhealthy: {health_status['reason']}")
            
            # Get unhealthy tasks
            unhealthy_tasks = get_unhealthy_tasks()
            
            if unhealthy_tasks:
                logger.info(f"Found {len(unhealthy_tasks)} unhealthy tasks")
                
                # Stop unhealthy tasks
                restart_result = restart_unhealthy_tasks(unhealthy_tasks)
                actions_taken.append(f"Stopped {restart_result['stopped_count']} unhealthy tasks")
                
                if restart_result['errors']:
                    logger.error(f"Errors during task restart: {restart_result['errors']}")
            else:
                # No specific unhealthy tasks found, force service update
                logger.info("No specific unhealthy tasks found, forcing service update")
                if force_service_update():
                    actions_taken.append("Forced service redeployment")
            
            # Send notification
            notification_message = f"""
Eden Health Monitor Alert

Service: {ECS_SERVICE}
Cluster: {ECS_CLUSTER}
Status: UNHEALTHY

Health Check Results:
- Running Tasks: {health_status['running_tasks']}/{health_status['desired_tasks']}
- Reason: {health_status['reason']}

Actions Taken:
{chr(10).join(['- ' + action for action in actions_taken])}

Timestamp: {datetime.utcnow().isoformat()}
API Gateway: {API_GATEWAY_URL}
"""
            
            send_sns_notification(
                subject=f"🚨 Eden Service Auto-Healing Triggered",
                message=notification_message
            )
            
            logger.info("✅ Auto-healing actions completed")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Auto-healing triggered',
                    'health_status': health_status,
                    'actions_taken': actions_taken,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        else:
            # Service is healthy
            logger.info("✅ Service is healthy, no action needed")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Service healthy, no action needed',
                    'health_status': health_status,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        
    except Exception as e:
        error_msg = f"Lambda execution failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # Try to send error notification
        if SNS_TOPIC_ARN:
            send_sns_notification(
                subject="🚨 Eden Health Monitor Error",
                message=f"Error in health monitor Lambda:\n\n{error_msg}\n\nTimestamp: {datetime.utcnow().isoformat()}"
            )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': error_msg,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
