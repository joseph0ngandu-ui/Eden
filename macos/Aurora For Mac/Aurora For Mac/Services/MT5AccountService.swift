import Combine
import Foundation

class MT5AccountService: ObservableObject {
    static let shared = MT5AccountService()
    
    @Published var accounts: [MT5Account] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let apiService = APIService.shared
    
    private init() {}
    
    // MARK: - Fetch Accounts
    
    func fetchAccounts() async throws {
        await MainActor.run { isLoading = true }
        
        do {
            let data = try await apiService.performRequest(endpoint: "/account/mt5", method: "GET")
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let accounts = try decoder.decode([MT5Account].self, from: data)
            
            await MainActor.run {
                self.accounts = accounts.sorted { $0.isPrimary && !$1.isPrimary }
                self.isLoading = false
                self.errorMessage = nil
            }
        } catch {
            await MainActor.run {
                self.isLoading = false
                self.errorMessage = error.localizedDescription
            }
            throw error
        }
    }
    
    // MARK: - Account Actions
    
    func addAccount(_ account: MT5AccountCreate) async throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(account)
        
        _ = try await apiService.performRequest(
            endpoint: "/account/mt5", method: "POST", body: data)
        try await fetchAccounts()
    }
    
    func updateAccount(id: Int, data: MT5AccountUpdate) async throws {
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(data)
        
        _ = try await apiService.performRequest(
            endpoint: "/account/mt5/\(id)", method: "PUT", body: jsonData)
        try await fetchAccounts()
    }
    
    func deleteAccount(id: Int) async throws {
        _ = try await apiService.performRequest(
            endpoint: "/account/mt5/\(id)", method: "DELETE")
        try await fetchAccounts()
    }
    
    func setPrimaryAccount(id: Int) async throws {
        _ = try await apiService.performRequest(
            endpoint: "/account/mt5/\(id)/primary", method: "PUT")
        try await fetchAccounts()
    }
}
