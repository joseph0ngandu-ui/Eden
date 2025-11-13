import json
import os
import boto3
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
ecs_client = boto3.client('ecs')
apigw_client = boto3.client('apigatewayv2')

# Environment variables
API_GATEWAY_ID = os.environ['API_GATEWAY_ID']
INTEGRATION_ID = os.environ['INTEGRATION_ID']
ECS_CLUSTER = os.environ['ECS_CLUSTER']
ECS_SERVICE = os.environ['ECS_SERVICE']
BACKEND_PORT = os.environ.get('BACKEND_PORT', '8000')


def get_latest_task_public_ip():
    """Fetch the public IP of the latest running ECS task."""
    try:
        # List tasks in the service
        logger.info(f"Fetching tasks for service: {ECS_SERVICE} in cluster: {ECS_CLUSTER}")
        response = ecs_client.list_tasks(
            cluster=ECS_CLUSTER,
            serviceName=ECS_SERVICE,
            desiredStatus='RUNNING'
        )
        
        if not response['taskArns']:
            logger.warning("No running tasks found")
            return None
        
        # Get task details
        task_arn = response['taskArns'][0]  # Get the first (latest) task
        logger.info(f"Describing task: {task_arn}")
        
        task_details = ecs_client.describe_tasks(
            cluster=ECS_CLUSTER,
            tasks=[task_arn]
        )
        
        if not task_details['tasks']:
            logger.error("Failed to describe task")
            return None
        
        task = task_details['tasks'][0]
        
        # Extract ENI ID from task attachments
        eni_id = None
        for attachment in task.get('attachments', []):
            if attachment['type'] == 'ElasticNetworkInterface':
                for detail in attachment['details']:
                    if detail['name'] == 'networkInterfaceId':
                        eni_id = detail['value']
                        break
        
        if not eni_id:
            logger.error("No ENI found in task attachments")
            return None
        
        logger.info(f"Found ENI: {eni_id}")
        
        # Get public IP from ENI
        ec2_client = boto3.client('ec2')
        eni_response = ec2_client.describe_network_interfaces(
            NetworkInterfaceIds=[eni_id]
        )
        
        if not eni_response['NetworkInterfaces']:
            logger.error("Failed to describe network interface")
            return None
        
        association = eni_response['NetworkInterfaces'][0].get('Association', {})
        public_ip = association.get('PublicIp')
        
        if public_ip:
            logger.info(f"Found public IP: {public_ip}")
            return public_ip
        else:
            logger.error("No public IP associated with ENI")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching task IP: {str(e)}")
        raise


def update_api_gateway_integration(new_ip):
    """Update the API Gateway integration with the new IP."""
    try:
        new_uri = f"http://{new_ip}:{BACKEND_PORT}/{{proxy}}"
        
        logger.info(f"Updating API Gateway integration to: {new_uri}")
        
        response = apigw_client.update_integration(
            ApiId=API_GATEWAY_ID,
            IntegrationId=INTEGRATION_ID,
            IntegrationUri=new_uri
        )
        
        logger.info(f"Successfully updated integration: {response['IntegrationId']}")
        logger.info(f"New integration URI: {response['IntegrationUri']}")
        
        return {
            'success': True,
            'integration_id': response['IntegrationId'],
            'new_uri': response['IntegrationUri']
        }
        
    except Exception as e:
        logger.error(f"Error updating API Gateway: {str(e)}")
        raise


def lambda_handler(event, context):
    """Main Lambda handler triggered by ECS task state changes."""
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Parse ECS event
        detail = event.get('detail', {})
        last_status = detail.get('lastStatus')
        desired_status = detail.get('desiredStatus')
        cluster_arn = detail.get('clusterArn', '')
        
        logger.info(f"Task status - Last: {last_status}, Desired: {desired_status}")
        
        # Only proceed if task is running
        if last_status != 'RUNNING':
            logger.info(f"Task not in RUNNING state (status: {last_status}), skipping update")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Task not running, no action taken',
                    'status': last_status
                })
            }
        
        # Verify it's our cluster
        if ECS_CLUSTER not in cluster_arn:
            logger.info(f"Event from different cluster, skipping")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Different cluster, no action taken'})
            }
        
        # Get new task IP
        new_ip = get_latest_task_public_ip()
        
        if not new_ip:
            logger.error("Failed to retrieve task IP")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to retrieve task IP'})
            }
        
        # Update API Gateway integration
        result = update_api_gateway_integration(new_ip)
        
        logger.info("✅ API Gateway integration updated successfully")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'API Gateway integration updated successfully',
                'timestamp': datetime.utcnow().isoformat(),
                'new_ip': new_ip,
                'integration_id': result['integration_id'],
                'new_uri': result['new_uri']
            })
        }
        
    except Exception as e:
        logger.error(f"❌ Lambda execution failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
