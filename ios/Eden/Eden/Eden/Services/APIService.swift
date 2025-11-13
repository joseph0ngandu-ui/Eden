//
//  APIService.swift
//  Eden
//
//  REST API service for Eden bot integration
//

import Foundation
import Combine

class APIService {
    static let shared = APIService()
    
    // MARK: - Configuration
    private let baseURL = APIEndpoints.baseURL
    private var authToken: String?
    private var cancellables = Set<AnyCancellable>()
    
    private init() {}
    
    // MARK: - Authentication
    func setAuthToken(_ token: String) {
        self.authToken = token
    }
    
    // MARK: - Request Builder
    private func buildRequest(url: URL, method: String = "GET", body: [String: Any]? = nil) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        
        // Add auth token if available
        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        if let body = body {
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }
        
        return request
    }
    
    // MARK: - Fetch Bot Status
    func fetchBotStatus(completion: @escaping (Result<BotStatus, Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.Bot.status) else {
            completion(.failure(APIError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: BotStatus.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { status in
                    completion(.success(status))
                }
            )
            .store(in: &cancellables)
    }
    
    // MARK: - Control Bot
    func controlBot(command: String, completion: @escaping (Result<String, Error>) -> Void) {
        let endpoint = command == "start" ? APIEndpoints.Bot.start : 
                      command == "stop" ? APIEndpoints.Bot.stop : APIEndpoints.Bot.pause
        guard let url = URL(string: endpoint) else {
            completion(.failure(APIError.invalidURL))
            return
        }
        
        let body = ["command": command]
        let request = buildRequest(url: url, method: "POST", body: body)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { _ in
                    completion(.success("Command '\(command)' sent successfully"))
                }
            )
            .store(in: &cancellables)
    }
    
    // MARK: - Fetch Positions
    func fetchPositions(completion: @escaping (Result<[Position], Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.Trades.open) else {
            completion(.failure(APIError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: [Position].self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { positions in
                    completion(.success(positions))
                }
            )
            .store(in: &cancellables)
    }
    
    // MARK: - Fetch Trades
    func fetchTrades(completion: @escaping (Result<[Trade], Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.Trades.history) else {
            completion(.failure(APIError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: [Trade].self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { trades in
                    completion(.success(trades))
                }
            )
            .store(in: &cancellables)
    }
}

// MARK: - API Errors
enum APIError: LocalizedError {
    case invalidURL
    case networkError
    case decodingError
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError:
            return "Network error occurred"
        case .decodingError:
            return "Failed to decode response"
        }
    }
}
