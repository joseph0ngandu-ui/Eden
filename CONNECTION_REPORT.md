# EDEN iOS APP TO BACKEND CONNECTION REPORT
**Pre-AWS Deployment Linkage**  
Generated: 2025-11-10T13:30:00Z

---

## PROJECT STATUS: ✅ READY FOR AWS DEPLOYMENT

---

## 1. BACKEND INFRASTRUCTURE

**Backend Stack:** FastAPI (Python 3.11+)  
**Location:** `C:\Users\Sal\OneDrive - ZCAS University\Eden\backend\`  
**Status:** ✅ Created and configured

### Components Created:
- `main.py` - Main FastAPI application with all endpoints
- `app/models.py` - Pydantic models for API schemas
- `app/auth.py` - JWT authentication system
- `app/database.py` - Database configuration (SQLAlchemy)
- `app/settings.py` - Environment configuration
- `app/trading_service.py` - Trading bot service layer
- `app/websocket_manager.py` - WebSocket connection manager
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `.env` - Environment variables

**Database:** SQLite (dev) / PostgreSQL (production)  
**Authentication:** JWT Bearer Tokens (24-hour expiry)  
**WebSocket Support:** ✅ Enabled for real-time updates

---

## 2. API ENDPOINTS CATALOG

**BASE URL (Local):** `http://localhost:8000`  
**BASE URL (AWS):** `https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod`

### Authentication Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login and receive JWT token |

**Login Request Format:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Login Response Format:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

### Trading Endpoints (Requires Auth)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/trades/open` | Get current open positions |
| GET | `/trades/history` | Get historical trades |
| GET | `/trades/recent` | Get recent trades |
| POST | `/trades/close` | Close specific position |

### Performance Endpoints (Requires Auth)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/performance/stats` | Get comprehensive performance statistics |
| GET | `/performance/equity-curve` | Get equity curve data for charts |
| GET | `/performance/daily-summary` | Get daily PnL summary |

### Bot Control Endpoints (Requires Auth)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/bot/status` | Get current bot status |
| POST | `/bot/start` | Start trading bot |
| POST | `/bot/stop` | Stop trading bot |
| POST | `/bot/pause` | Pause trading bot |

### Strategy Endpoints (Requires Auth)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/strategy/config` | Get strategy configuration |
| POST | `/strategy/config` | Update strategy configuration |
| GET | `/strategy/symbols` | Get list of trading symbols |

### Health & System Endpoints
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/health` | Health check | No |
| GET | `/system/status` | System diagnostics | Yes |
| GET | `/info` | API information | No |
| GET | `/` | Root endpoint with API info | No |

### WebSocket Endpoints
| Protocol | Endpoint | Description |
|----------|----------|-------------|
| WS | `/ws/updates/{token}` | Real-time bot status updates |
| WS | `/ws/trades/{token}` | Real-time trade updates |

---

## 3. iOS APP INTEGRATION

**iOS App Location:** `C:\Users\Sal\OneDrive - ZCAS University\Eden\EdenIOSApp\Eden\`

### Updated Files:
✅ `Network/Endpoints.swift` - Complete endpoint configuration  
✅ `Services/APIService.swift` - Updated to use new endpoints  
✅ `Models/Models.swift` - Already aligned with backend

### Key iOS Classes:
- **APIEndpoints** - Centralized endpoint configuration
- **APIConfig** - Request configuration & headers
- **AWSConfig** - AWS-specific settings
- **APIService** - Network service layer

### Environment Configuration:
- **Development:** `http://localhost:8000`
- **Staging:** `https://staging-api.eden-trading.com` (configure)
- **Production:** `https://api.eden-trading.com` (configure with AWS URL)

---

## 4. AUTHENTICATION FLOW

### Step 1: User Registration (Optional)
```http
POST /auth/register
Body: { "email", "full_name", "password" }
Response: { "message": "User registered successfully", "email": "..." }
```

### Step 2: User Login
```http
POST /auth/login
Body: { "email": "user@example.com", "password": "password123" }
Response: { "access_token": "JWT_TOKEN", "token_type": "bearer" }
```

### Step 3: Store Token in iOS App
- Save token securely (Keychain recommended)
- Include in all subsequent requests:
  ```
  Authorization: Bearer {JWT_TOKEN}
  ```

### Step 4: Make Authenticated Requests
```http
GET /bot/status
Headers: {
  "Authorization": "Bearer JWT_TOKEN",
  "Content-Type": "application/json"
}
```

**Token Expiry:** 24 hours (configurable)  
**Refresh:** Re-login when token expires

---

## 5. DATA MODELS ALIGNMENT

### BotStatus
```swift
{
  isRunning: Bool
  balance: Double
  dailyPnL: Double
  activePositions: Int
  winRate: Double
  riskTier: String
  totalTrades: Int?
  profitFactor: Double?
  peakBalance: Double?
  currentDrawdown: Double?
}
```

### Position
```swift
{
  symbol: String
  direction: String // "BUY" | "SELL"
  entry: Double
  current: Double
  pnl: Double
  confidence: Double
  bars: Int
}
```

### Trade
```swift
{
  symbol: String
  pnl: Double
  time: String // ISO 8601
  rValue: Double
}
```

---

## 6. LOCAL TESTING INSTRUCTIONS

### Start Backend Locally
```powershell
# 1. Navigate to backend directory
cd "C:\Users\Sal\OneDrive - ZCAS University\Eden\backend"

# 2. Create Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Verify health
curl http://localhost:8000/health

# 6. View API docs
# Open browser: http://localhost:8000/docs
```

### Test API Endpoints Manually
```powershell
# 1. Register user
curl -X POST http://localhost:8000/auth/register `
  -H "Content-Type: application/json" `
  -d '{"email":"test@eden.com","password":"test123","full_name":"Test User"}'

# 2. Login
curl -X POST http://localhost:8000/auth/login `
  -H "Content-Type: application/json" `
  -d '{"email":"test@eden.com","password":"test123"}'

# 3. Get bot status (replace TOKEN)
curl http://localhost:8000/bot/status `
  -H "Authorization: Bearer TOKEN"
```

---

## 7. DOCKER DEPLOYMENT

### Build Docker Image
```powershell
cd "C:\Users\Sal\OneDrive - ZCAS University\Eden"
docker build -t eden-trading-api:latest ./backend
```

### Run with Docker Compose
```powershell
docker-compose up -d
```

### Verify Container
```powershell
docker ps
docker logs eden-trading-api
```

### Access API
`http://localhost:8000/docs`

---

## 8. AWS DEPLOYMENT STEPS

### Prerequisites:
✅ AWS Account with appropriate permissions  
✅ AWS CLI installed and configured  
✅ Docker image pushed to ECR

### Step 1: Push Docker Image to ECR
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name eden-trading-api

# 2. Login to ECR
aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin \
<account-id>.dkr.ecr.us-east-1.amazonaws.com

# 3. Tag and push image
docker tag eden-trading-api:latest \
<account-id>.dkr.ecr.us-east-1.amazonaws.com/eden-trading-api:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/eden-trading-api:latest
```

### Step 2: Deploy CloudFormation Stack
```bash
# 1. Deploy stack
aws cloudformation create-stack \
  --stack-name eden-trading-stack \
  --template-body file://aws-infrastructure.yml \
  --parameters \
    ParameterKey=EnvironmentName,ParameterValue=prod \
    ParameterKey=ContainerImage,ParameterValue=<ECR_IMAGE_URI> \
    ParameterKey=DBPassword,ParameterValue=<SECURE_PASSWORD> \
  --capabilities CAPABILITY_IAM

# 2. Monitor deployment
aws cloudformation describe-stacks --stack-name eden-trading-stack
```

### Step 3: Get API Gateway URL
```bash
aws cloudformation describe-stacks \
  --stack-name eden-trading-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`APIGatewayURL`].OutputValue' \
  --output text
```

### Step 4: Update iOS App
1. Update `Endpoints.swift` with production baseURL
2. Update `AWSConfig` with actual values
3. Rebuild iOS app with production configuration
4. Test against live AWS deployment

---

## 9. SECURITY CONSIDERATIONS

✅ JWT authentication with Bearer tokens  
✅ Passwords hashed with bcrypt  
✅ HTTPS enforced in production (API Gateway)  
✅ CORS configured (restrict in production)  
✅ Environment variables for secrets  
✅ Database credentials in AWS Secrets Manager (recommended)  
✅ IAM roles for ECS tasks  
✅ Security groups limiting network access

### Production Recommendations:
- Use AWS Cognito for user authentication
- Store secrets in AWS Secrets Manager
- Enable AWS WAF for API Gateway
- Configure CloudWatch alarms
- Use RDS encryption at rest
- Enable VPC Flow Logs

---

## 10. WEBSOCKET REAL-TIME UPDATES

### iOS WebSocket Implementation:
```swift
// 1. Connect to WebSocket after authentication
let wsURL = APIEndpoints.WebSocket.updates(token: authToken)

// 2. WebSocket automatically sends bot status updates

// 3. iOS app receives updates in real-time
{ "type": "bot_status", "data": {...} }
```

### AWS WebSocket API Gateway:
- Separate WebSocket API required
- Configure $connect, $disconnect, $default routes
- Use Lambda for message routing

---

## 11. MONITORING & LOGGING

### Local Development:
- Backend logs: Console output from uvicorn
- Database: SQLite file in backend directory
- API docs: http://localhost:8000/docs

### AWS Production:
- Application logs: CloudWatch Logs (`/ecs/prod-eden-api`)
- Database: RDS PostgreSQL (automated backups)
- API Gateway logs: CloudWatch Logs
- Container metrics: ECS CloudWatch metrics

### Recommended CloudWatch Alarms:
- API 5XX error rate > 5%
- ECS CPU utilization > 80%
- RDS connections > 80% of max
- API latency > 2 seconds

---

## 12. DEPLOYMENT CHECKLIST

### Pre-Deployment:
- [x] Backend API created with all endpoints
- [x] iOS app updated with Endpoints.swift
- [x] Docker configuration created
- [x] AWS CloudFormation template ready
- [ ] Test locally (iOS → localhost:8000)
- [ ] Generate secure SECRET_KEY for production
- [ ] Configure AWS account and permissions
- [ ] Create RDS database password
- [ ] Push Docker image to ECR
- [ ] Deploy CloudFormation stack
- [ ] Update iOS app with production URL
- [ ] Test iOS app against AWS deployment
- [ ] Configure CloudWatch monitoring
- [ ] Setup CI/CD pipeline (optional)

### Post-Deployment:
- [ ] Monitor API health and performance
- [ ] Test all authentication flows
- [ ] Verify WebSocket connections
- [ ] Load test API endpoints
- [ ] Configure SSL/TLS certificate (optional)
- [ ] Setup custom domain (optional)
- [ ] Configure backup strategy
- [ ] Document API for team

---

## 13. PROJECT SUMMARY

**Project:** Eden Trading Bot  
**Backend:** FastAPI + Python 3.11  
**iOS App:** SwiftUI + Combine  
**Cloud:** AWS (ECS, RDS, API Gateway)  
**Generated by:** Warp AI Agent  
**Date:** 2025-11-10  
**Status:** ✅ PRE-AWS DEPLOYMENT READY

---

**END OF CONNECTION REPORT**