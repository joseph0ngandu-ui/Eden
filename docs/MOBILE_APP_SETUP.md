# 📱 Eden Mobile App Connection Guide

## ✅ Server Ready for Mobile App!

Your Eden trading bot backend is now running with HTTPS and ready for your iOS/Android app.

---

## 🌐 Connection Details

### API Endpoints

**Public Access (from anywhere):**
```
https://13.50.226.20:8443
```

**Local Network Access:**
```
https://172.31.11.145:8443
```

### Login Credentials
```
Email: admin@eden.com
Password: admin123
```

---

## 📱 Mobile App Configuration

### Step 1: Configure API URL in Your App

In your Eden iOS/Android app settings, enter:
```
API URL: https://13.50.226.20:8443
```

### Step 2: Accept Certificate Warning

On first connection, you'll see a certificate warning because we're using a self-signed SSL certificate. This is normal and safe for this setup.

**iOS:**
- Tap "Continue" or "Accept"
- Go to Settings → General → About → Certificate Trust Settings
- Enable trust for "Eden Trading Bot API"

**Android:**
- Tap "Accept" or "Continue Anyway"
- The app will remember your choice

### Step 3: Login

Use the credentials above to log in.

---

## 🔒 Security Notes

- **HTTPS Enabled:** All communication is encrypted
- **Self-Signed Certificate:** Prevents man-in-the-middle attacks
- **Firewall:** Ensure port 8443 is open in AWS Security Group

---

## 🛠️ AWS Security Group Configuration

Make sure your EC2 Security Group allows incoming traffic on port 8443:

1. Go to EC2 Dashboard
2. Select your instance
3. Click Security tab → Security Groups
4. Add Inbound Rule:
   - Type: Custom TCP
   - Port: 8443
   - Source: 0.0.0.0/0 (or your specific IP)

---

## 📊 Available API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Get JWT token

### Trading
- `GET /trades/open` - View open positions
- `GET /trades/history` - Trade history
- `GET /trades/recent?days=7` - Recent trades
- `GET /bot/status` - Bot status

### Account Management
- `POST /accounts/add` - Add MT5 account
- `GET /accounts` - List accounts
- `PUT /accounts/{id}` - Update account
- `DELETE /accounts/{id}` - Remove account

### Documentation
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

---

## 🧪 Test Connection

### From Browser
Visit: `https://13.50.226.20:8443/docs`

Accept the certificate warning and you'll see the API documentation.

### From Command Line
```bash
curl -k https://13.50.226.20:8443/docs
```

### From Python
```python
import requests
requests.packages.urllib3.disable_warnings()
response = requests.get('https://13.50.226.20:8443/docs', verify=False)
print(response.status_code)  # Should be 200
```

---

## 🔄 Backend Status

**Running:** Yes  
**Protocol:** HTTPS  
**Port:** 8443  
**Database:** SQLite (eden_trading.db)  
**Logs:** C:\Users\Administrator\Eden\logs\

---

## 📞 Troubleshooting

### Can't Connect from Mobile App?

1. **Check AWS Security Group:**
   - Port 8443 must be open
   - Source: 0.0.0.0/0 or your IP

2. **Check Backend is Running:**
   ```powershell
   Get-Process python | Where-Object {$_.CommandLine -like "*uvicorn*"}
   ```

3. **Restart Backend:**
   ```batch
   C:\Users\Administrator\Eden\backend\START_BACKEND_HTTPS.bat
   ```

4. **Check Firewall:**
   ```powershell
   netsh advfirewall firewall show rule name="Eden HTTPS"
   ```

### Certificate Issues?

The certificate is self-signed and valid for 5 years. Your app must accept it explicitly on first connection.

### Connection Timeout?

- Ensure your EC2 instance is running
- Check Security Group allows port 8443
- Verify backend is running with command above

---

## ✨ Next Steps

1. Open your Eden mobile app
2. Enter API URL: `https://13.50.226.20:8443`
3. Accept certificate warning
4. Login with credentials above
5. View bot status and trades!

---

**Status:** 🟢 READY FOR MOBILE APP  
**Last Updated:** 2025-11-11  
**API Version:** 1.0.0
