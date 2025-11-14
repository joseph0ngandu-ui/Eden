(function(){
  // EDEN SERVER CONFIGURATION
  // Update these values to match your server
  window.EDEN_CONFIG = {
    // Your server URL (no trailing slash)
    // For InfinityFree deployment, this will be set automatically to use api.php
    API_URL: 'https://edenbot.duckdns.org:8443', // Live Eden backend with Let's Encrypt
    
    // Authentication
    TOKEN_KEY: 'eden_token',
    REFRESH_TOKEN_KEY: 'eden_refresh_token',
    
    // Request timeout in milliseconds
    TIMEOUT: 10000,
    
    // Enable this for debugging
    DEBUG: false
  };
})();