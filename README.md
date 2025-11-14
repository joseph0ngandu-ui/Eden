# Eden Web Dashboard

A web dashboard for Eden with full server connectivity. The app is a single-page application (SPA) that automatically detects your deployment environment and connects to your server or uses the included PHP proxy for InfinityFree hosting.

## Structure

- `index.html` — main SPA container
- `assets/css/styles.css` — styles
- `assets/js/storage.js` — token-based authentication storage
- `assets/js/deploy-config.js` — automatic deployment detection
- `assets/js/config.js` — server configuration (can override deploy-config)
- `assets/js/api.js` — server-connected API layer
- `assets/js/router.js` — tiny hash router
- `assets/js/app.js` — views and event wiring
- `api.php` — PHP proxy for InfinityFree deployment

## Run locally

Just open `index.html` in a browser, or use a static server:

```bash
# Python
python -m http.server 8080
# or Node
npx serve .
```

## Server Integration

Your server needs to expose the following REST endpoints:

```
POST /auth/login      - Login with email/password
POST /auth/logout     - Logout
GET  /tasks          - List tasks
POST /tasks          - Create task
PUT  /tasks/:id      - Update task
DELETE /tasks/:id    - Delete task
GET  /notifications  - List notifications
PUT  /notifications/:id - Mark notification as read
GET  /messages       - List messages
POST /messages       - Send message
GET  /insights       - Get analytics data
GET  /settings       - Get user settings
PUT  /settings       - Update settings
```

Your server should return JSON responses with this structure:

```json
// Login response
{
  "user": { "id": "...", "name": "...", "email": "..." },
  "token": "jwt-token-here",
  "refreshToken": "optional-refresh-token"
}

// Error responses
{ "message": "Error description" }
```

## Deploy to InfinityFree

1. Create a new site in InfinityFree and open the File Manager (or use FTP).
2. Update the API endpoint in `assets/js/config.js`:
   ```
   API_URL: 'https://edenbot.duckdns.org:8443'  // Live Eden backend with Let's Encrypt
   ```
3. Upload the entire project folder contents to the document root (usually `htdocs/`).
4. Ensure `index.html` and `api.php` are in the root of `htdocs/`.
5. Visit your site URL and the dashboard should connect to your server.

The dashboard includes a PHP proxy (`api.php`) that handles CORS issues and forwards requests to your server. When deployed on InfinityFree, the configuration will automatically detect this and use the proxy.

## Deploy to other hosting

The codebase is compatible with any hosting that can serve static files. Simply update the `API_URL` in `assets/js/config.js` to point to your server and upload the files.

## Git setup

```bash
git init
git add .
git commit -m "feat: Eden dashboard with server integration"
# Create a new repo on your Git host and add it as origin:
# git remote add origin https://github.com/your-username/eden-dashboard.git
# git branch -M main
# git push -u origin main
```

## Notes

- This project uses only vanilla HTML/CSS/JS for maximum compatibility with hosting platforms.
- The dashboard automatically detects InfinityFree hosting and configures itself to use the PHP proxy.
- Authentication tokens are stored in localStorage and automatically included in all API requests.
- The PHP proxy will forward authentication headers and responses between the browser and your server.
