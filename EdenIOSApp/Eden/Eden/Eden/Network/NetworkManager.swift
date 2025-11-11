//
//  NetworkManager.swift
//  Eden
//
//  Network manager with SSL certificate trust for self-signed certificates
//  IMPORTANT: Only use for development/testing with known servers
//

import Foundation

class NetworkManager: NSObject {
    static let shared = NetworkManager()
    
    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        return URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }()
    
    private override init() {
        super.init()
    }
    
    /// Perform a data task with SSL bypass for self-signed certificates
    func dataTask(with request: URLRequest, completion: @escaping (Data?, URLResponse?, Error?) -> Void) -> URLSessionDataTask {
        return session.dataTask(with: request, completionHandler: completion)
    }
    
    /// Perform a data task with URL
    func dataTask(with url: URL, completion: @escaping (Data?, URLResponse?, Error?) -> Void) -> URLSessionDataTask {
        return session.dataTask(with: url, completionHandler: completion)
    }
}

// MARK: - URLSessionDelegate
extension NetworkManager: URLSessionDelegate {
    
    /// Handle SSL certificate validation
    /// WARNING: This bypasses certificate validation for self-signed certificates
    /// Only use this for development/testing with known servers
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge, completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        
        // Check if this is a server trust challenge
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let serverTrust = challenge.protectionSpace.serverTrust else {
            completionHandler(.performDefaultHandling, nil)
            return
        }
        
        // For development: Accept self-signed certificates from our known server
        // In production with valid certificates, remove this or make it configurable
        let host = challenge.protectionSpace.host
        
        // Only bypass SSL for our specific server
        if host == "13.50.226.20" || host == "localhost" || host == "127.0.0.1" {
            let credential = URLCredential(trust: serverTrust)
            completionHandler(.useCredential, credential)
        } else {
            // For other hosts, use default handling
            completionHandler(.performDefaultHandling, nil)
        }
    }
}

// MARK: - Convenience Methods
extension NetworkManager {
    
    /// Perform GET request
    func get(url: URL, headers: [String: String]? = nil, completion: @escaping (Result<Data, Error>) -> Void) {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        
        headers?.forEach { key, value in
            request.setValue(value, forHTTPHeaderField: key)
        }
        
        let task = dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NetworkError.noData))
                return
            }
            
            completion(.success(data))
        }
        
        task.resume()
    }
    
    /// Perform POST request
    func post(url: URL, body: Data?, headers: [String: String]? = nil, completion: @escaping (Result<Data, Error>) -> Void) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = body
        
        headers?.forEach { key, value in
            request.setValue(value, forHTTPHeaderField: key)
        }
        
        let task = dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NetworkError.noData))
                return
            }
            
            completion(.success(data))
        }
        
        task.resume()
    }
}

// MARK: - Network Errors
enum NetworkError: LocalizedError {
    case noData
    case invalidResponse
    case serverError(statusCode: Int)
    
    var errorDescription: String? {
        switch self {
        case .noData:
            return "No data received from server"
        case .invalidResponse:
            return "Invalid response from server"
        case .serverError(let statusCode):
            return "Server error: HTTP \(statusCode)"
        }
    }
}
