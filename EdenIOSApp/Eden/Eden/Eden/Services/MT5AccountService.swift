//
//  MT5AccountService.swift
//  Eden
//
//  MT5 Account API service - syncs with backend
//

import Foundation
import Combine

// MARK: - MT5 Account Models

struct MT5AccountData: Codable, Identifiable {
    let id: Int
    let user_id: Int
    let account_number: String
    let account_name: String?
    let broker: String?
    let server: String?
    let is_active: Bool
    let is_primary: Bool
    let created_at: String
    let updated_at: String
    let last_synced: String?
}

struct MT5AccountCreate: Codable {
    let account_number: String
    let account_name: String?
    let broker: String?
    let server: String?
    let password: String?
    let is_primary: Bool
}

struct MT5AccountUpdate: Codable {
    let account_name: String?
    let broker: String?
    let server: String?
    let password: String?
    let is_primary: Bool?
    let is_active: Bool?
}

// MARK: - MT5 Account Service

class MT5AccountService {
    static let shared = MT5AccountService()
    
    private let baseURL = APIEndpoints.baseURL
    private var authToken: String?
    private var cancellables = Set<AnyCancellable>()
    
    private init() {}
    
    // MARK: - Configuration
    
    func setAuthToken(_ token: String) {
        self.authToken = token
    }
    
    private func buildRequest(url: URL, method: String = "GET", body: Encodable? = nil) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        
        // Add auth token
        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        // Add body if present
        if let body = body {
            request.httpBody = try? JSONEncoder().encode(body)
        }
        
        return request
    }
    
    // MARK: - API Methods
    
    /// Get all MT5 accounts for current user
    func getAccounts(completion: @escaping (Result<[MT5AccountData], Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.MT5Account.list) else {
            completion(.failure(MT5AccountError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: [MT5AccountData].self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { accounts in
                    completion(.success(accounts))
                }
            )
            .store(in: &cancellables)
    }
    
    /// Get primary MT5 account
    func getPrimaryAccount(completion: @escaping (Result<MT5AccountData, Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.MT5Account.primary) else {
            completion(.failure(MT5AccountError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: MT5AccountData.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { account in
                    completion(.success(account))
                }
            )
            .store(in: &cancellables)
    }
    
    /// Create new MT5 account
    func createAccount(_ accountData: MT5AccountCreate, completion: @escaping (Result<MT5AccountData, Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.MT5Account.create) else {
            completion(.failure(MT5AccountError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url, method: "POST", body: accountData)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: MT5AccountData.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { account in
                    // Also save to UserDefaults for offline access
                    self.saveToUserDefaults(account)
                    completion(.success(account))
                }
            )
            .store(in: &cancellables)
    }
    
    /// Update existing MT5 account
    func updateAccount(accountId: Int, updateData: MT5AccountUpdate, completion: @escaping (Result<MT5AccountData, Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.MT5Account.update(accountId: accountId)) else {
            completion(.failure(MT5AccountError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url, method: "PUT", body: updateData)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: MT5AccountData.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { result in
                    if case .failure(let error) = result {
                        completion(.failure(error))
                    }
                },
                receiveValue: { account in
                    // Update UserDefaults
                    self.saveToUserDefaults(account)
                    completion(.success(account))
                }
            )
            .store(in: &cancellables)
    }
    
    /// Delete MT5 account (soft delete)
    func deleteAccount(accountId: Int, completion: @escaping (Result<Void, Error>) -> Void) {
        guard let url = URL(string: APIEndpoints.MT5Account.delete(accountId: accountId)) else {
            completion(.failure(MT5AccountError.invalidURL))
            return
        }
        
        let request = buildRequest(url: url, method: "DELETE")
        
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
                    completion(.success(()))
                }
            )
            .store(in: &cancellables)
    }
    
    // MARK: - Local Storage (for offline access)
    
    private func saveToUserDefaults(_ account: MT5AccountData) {
        UserDefaults.standard.set(account.account_number, forKey: "mt5AccountNumber")
        UserDefaults.standard.set(account.account_name, forKey: "mt5AccountName")
        UserDefaults.standard.set(account.broker, forKey: "mt5Broker")
        UserDefaults.standard.set(account.server, forKey: "mt5Server")
        UserDefaults.standard.set(account.is_primary, forKey: "mt5IsPrimary")
    }
    
    func loadFromUserDefaults() -> (number: String, name: String, broker: String, server: String)? {
        guard let number = UserDefaults.standard.string(forKey: "mt5AccountNumber"),
              !number.isEmpty else {
            return nil
        }
        
        let name = UserDefaults.standard.string(forKey: "mt5AccountName") ?? ""
        let broker = UserDefaults.standard.string(forKey: "mt5Broker") ?? ""
        let server = UserDefaults.standard.string(forKey: "mt5Server") ?? ""
        
        return (number, name, broker, server)
    }
    
    /// Sync with backend and update local storage
    func syncWithBackend(completion: @escaping (Result<MT5AccountData?, Error>) -> Void) {
        getPrimaryAccount { result in
            switch result {
            case .success(let account):
                self.saveToUserDefaults(account)
                completion(.success(account))
            case .failure(let error):
                // If no primary account or error, try to load from local storage
                print("⚠️ Couldn't sync with backend: \(error.localizedDescription)")
                completion(.success(nil))
            }
        }
    }
}

// MARK: - Errors

enum MT5AccountError: LocalizedError {
    case invalidURL
    case networkError
    case decodingError
    case unauthorized
    case accountNotFound
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError:
            return "Network error occurred"
        case .decodingError:
            return "Failed to decode response"
        case .unauthorized:
            return "Unauthorized - please login"
        case .accountNotFound:
            return "MT5 account not found"
        }
    }
}
