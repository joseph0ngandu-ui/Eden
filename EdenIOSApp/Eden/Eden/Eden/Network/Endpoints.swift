//
//  Endpoints.swift
//  Eden
//
//  API endpoint configuration for Eden Trading Bot backend
//  Auto-generated for AWS deployment
//

import Foundation

/// API Endpoints Configuration
struct APIEndpoints {
    
    // MARK: - Base Configuration
    
    /// Base URL for API (Update this with AWS API Gateway URL after deployment)
    static let baseURL = ProcessInfo.processInfo.environment["API_BASE_URL"] ?? "http://localhost:8000"
    
    /// WebSocket base URL
    static let wsBaseURL = baseURL.replacingOccurrences(of: "http", with: "ws")
    
    // MARK: - Authentication Endpoints
    
    struct Auth {
        static let register = "\(baseURL)/auth/register"
        static let login = "\(baseURL)/auth/login"
    }
    
    // MARK: - Trading Endpoints
    
    struct Trades {
        static let open = "\(baseURL)/trades/open"
        static let history = "\(baseURL)/trades/history"
        static let recent = "\(baseURL)/trades/recent"
        static let close = "\(baseURL)/trades/close"
        
        /// Get trade history with limit
        static func history(limit: Int) -> String {
            return "\(baseURL)/trades/history?limit=\(limit)"
        }
        
        /// Get recent trades by days
        static func recent(days: Int) -> String {
            return "\(baseURL)/trades/recent?days=\(days)"
        }
    }
    
    // MARK: - Performance Endpoints
    
    struct Performance {
        static let stats = "\(baseURL)/performance/stats"
        static let equityCurve = "\(baseURL)/performance/equity-curve"
        static let dailySummary = "\(baseURL)/performance/daily-summary"
    }
    
    // MARK: - Bot Control Endpoints
    
    struct Bot {
        static let status = "\(baseURL)/bot/status"
        static let start = "\(baseURL)/bot/start"
        static let stop = "\(baseURL)/bot/stop"
        static let pause = "\(baseURL)/bot/pause"
    }
    
    // MARK: - Strategy Endpoints
    
    struct Strategy {
        static let config = "\(baseURL)/strategy/config"
        static let symbols = "\(baseURL)/strategy/symbols"
    }
    
    // MARK: - MT5 Account Endpoints
    
    struct MT5Account {
        /// Get all MT5 accounts
        static let list = "\(baseURL)/account/mt5"
        
        /// Get primary MT5 account
        static let primary = "\(baseURL)/account/mt5/primary"
        
        /// Create new MT5 account
        static let create = "\(baseURL)/account/mt5"
        
        /// Update MT5 account by ID
        static func update(accountId: Int) -> String {
            return "\(baseURL)/account/mt5/\(accountId)"
        }
        
        /// Delete MT5 account by ID
        static func delete(accountId: Int) -> String {
            return "\(baseURL)/account/mt5/\(accountId)"
        }
    }
    
    // MARK: - Health & System Endpoints
    
    struct System {
        static let health = "\(baseURL)/health"
        static let status = "\(baseURL)/system/status"
        static let info = "\(baseURL)/info"
    }
    
    // MARK: - WebSocket Endpoints
    
    struct WebSocket {
        /// Real-time bot status updates
        static func updates(token: String) -> String {
            return "\(wsBaseURL)/ws/updates/\(token)"
        }
        
        /// Real-time trade updates
        static func trades(token: String) -> String {
            return "\(wsBaseURL)/ws/trades/\(token)"
        }
    }
    
    // MARK: - Helper Methods
    
    /// Update base URL (useful for switching between dev/staging/prod)
    static func updateBaseURL(_ newURL: String) {
        // Note: This would require refactoring to use a mutable storage
        // For now, use environment variables or configuration files
    }
    
    /// Check if using local development server
    static var isLocalDevelopment: Bool {
        return baseURL.contains("localhost") || baseURL.contains("127.0.0.1")
    }
    
    /// Check if using AWS deployment
    static var isAWSDeployment: Bool {
        return baseURL.contains("amazonaws.com") || baseURL.contains("aws")
    }
}

// MARK: - Environment Configuration

extension APIEndpoints {
    
    /// Environment types
    enum Environment: String {
        case development = "development"
        case staging = "staging"
        case production = "production"
        
        var baseURL: String {
            switch self {
            case .development:
                return "http://localhost:8000"
            case .staging:
                return "https://staging-api.eden-trading.com"  // Update with actual staging URL
            case .production:
                return "https://api.eden-trading.com"  // Update with AWS API Gateway URL
            }
        }
    }
    
    /// Current environment (read from Info.plist or environment variable)
    static var currentEnvironment: Environment {
        if let envString = ProcessInfo.processInfo.environment["EDEN_ENV"],
           let env = Environment(rawValue: envString) {
            return env
        }
        
        #if DEBUG
        return .development
        #else
        return .production
        #endif
    }
}

// MARK: - API Request Configuration

struct APIConfig {
    
    /// Default request timeout
    static let timeout: TimeInterval = 30.0
    
    /// WebSocket timeout
    static let wsTimeout: TimeInterval = 60.0
    
    /// Maximum retry attempts
    static let maxRetries: Int = 3
    
    /// Common headers
    static var defaultHeaders: [String: String] {
        return [
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Client-Version": appVersion,
            "X-Client-Platform": "iOS"
        ]
    }
    
    /// App version
    static var appVersion: String {
        return Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0"
    }
    
    /// Authorization header with Bearer token
    static func authHeader(token: String) -> [String: String] {
        var headers = defaultHeaders
        headers["Authorization"] = "Bearer \(token)"
        return headers
    }
}

// MARK: - AWS Configuration (For Production)

struct AWSConfig {
    
    /// AWS Region
    static let region = "us-east-1"
    
    /// API Gateway URL (Update after AWS deployment)
    static let apiGatewayURL = "https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod"
    
    /// WebSocket API Gateway URL
    static let wsApiGatewayURL = "wss://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod"
    
    /// Cognito Configuration (Optional)
    struct Cognito {
        static let userPoolId = "us-east-1_xxxxxxxxx"
        static let clientId = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
        static let identityPoolId = "us-east-1:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    }
}

// MARK: - Endpoint Builder Helper

struct EndpointBuilder {
    
    /// Build URL with query parameters
    static func build(_ baseURL: String, params: [String: String]) -> String {
        guard !params.isEmpty else { return baseURL }
        
        let queryItems = params.map { key, value in
            "\(key)=\(value.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? value)"
        }.joined(separator: "&")
        
        return "\(baseURL)?\(queryItems)"
    }
    
    /// Build URL with path components
    static func build(_ baseURL: String, path: [String]) -> String {
        let pathString = path.joined(separator: "/")
        return "\(baseURL)/\(pathString)"
    }
}
