(function(){
  // This script runs before the main config to detect deployment environment
  // and automatically configure the API URL
  
  // Determine if we're on InfinityFree or another static host
  const isInfinityFree = window.location.hostname.includes('.infinityfree.com') || 
                         window.location.hostname.includes('epizy.com') ||
                         window.location.pathname.includes('htdocs');
  
  // Set API URL based on environment
  const API_URL = isInfinityFree 
    ? window.location.origin + '/api.php'
    : 'https://your-eden-backend.com/api'; // Replace with your DNS backend URL
  
  // Update the global config
  window.EDEN_CONFIG = {
    API_URL,
    
    // Authentication
    TOKEN_KEY: 'eden_token',
    REFRESH_TOKEN_KEY: 'eden_refresh_token',
    
    // Request timeout in milliseconds
    TIMEOUT: 10000,
    
    // Enable this for debugging
    DEBUG: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  };
  
  // Show a notification in the auth dialog so users know which mode is active
  const authMessage = document.getElementById('authMessage');
  if (authMessage) {
    if (isInfinityFree) {
      authMessage.textContent = 'Deployed on InfinityFree with PHP proxy';
    } else if (API_URL.includes('your-server.com')) {
      authMessage.textContent = 'Demo mode - update API_URL in config.js to connect to your server';
    } else {
      authMessage.textContent = 'Connected to server: ' + API_URL;
    }
  }
})();