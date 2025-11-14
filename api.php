<?php
// Eden Dashboard API Proxy for InfinityFree
// This file handles CORS and proxies requests to your server

// Enable CORS for all origins (InfinityFree requirement)
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Authorization");

// Handle preflight OPTIONS requests
if ($_SERVER['REQUEST_METHOD'] == 'OPTIONS') {
    http_response_code(200);
    exit;
}

// Load configuration
$config_file = __DIR__ . '/assets/js/config.js';
$config_content = is_file($config_file) ? file_get_contents($config_file) : '';

// Extract API_URL from config (simple extraction)
preg_match('/API_URL:\s*[\'"]([^\'"]+)/', $config_content, $matches);
$api_url = isset($matches[1]) ? $matches[1] : '';

if (empty($api_url)) {
    http_response_code(500);
    echo json_encode(['error' => 'API URL not configured']);
    exit;
}

// Add the current path to the API URL
$path = $_SERVER['REQUEST_URI'];
if (strpos($path, '/api.php') === 0) {
    $path = substr($path, 8); // Remove '/api.php'
    if (empty($path)) $path = '/'; // Ensure path starts with /
}

$url = $api_url . $path;
$method = $_SERVER['REQUEST_METHOD'];

$headers = [];
$headers[] = 'Content-Type: application/json';

// Forward Authorization header if present
if (isset($_SERVER['HTTP_AUTHORIZATION'])) {
    $headers[] = 'Authorization: ' . $_SERVER['HTTP_AUTHORIZATION'];
}

// Prepare request
$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $method);
curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

// Add request body for POST/PUT requests
if (in_array($method, ['POST', 'PUT']) && !empty($_SERVER['CONTENT_TYPE']) && strpos($_SERVER['CONTENT_TYPE'], 'application/json') === 0) {
    $body = file_get_contents('php://input');
    curl_setopt($ch, CURLOPT_POSTFIELDS, $body);
}

// Execute request
$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($error) {
    http_response_code(500);
    echo json_encode(['error' => 'Proxy error: ' . $error]);
    exit;
}

// Return response with proper status code
http_response_code($http_code);
echo $response;
?>