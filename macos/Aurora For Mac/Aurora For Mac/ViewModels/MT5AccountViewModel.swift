import Combine
import Foundation
import SwiftUI

class MT5AccountViewModel: ObservableObject {
    @Published var accounts: [MT5Account] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let accountService = MT5AccountService.shared
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        accountService.$accounts
            .receive(on: DispatchQueue.main)
            .assign(to: &$accounts)
        
        accountService.$isLoading
            .receive(on: DispatchQueue.main)
            .assign(to: &$isLoading)
        
        accountService.$errorMessage
            .receive(on: DispatchQueue.main)
            .assign(to: &$errorMessage)
    }
    
    func loadAccounts() async {
        do {
            try await accountService.fetchAccounts()
        } catch {
            // Error handled in service and published
        }
    }
    
    func addAccount(_ account: MT5AccountCreate) async -> Bool {
        do {
            try await accountService.addAccount(account)
            return true
        } catch {
            return false
        }
    }
    
    func updateAccount(id: Int, data: MT5AccountUpdate) async -> Bool {
        do {
            try await accountService.updateAccount(id: id, data: data)
            return true
        } catch {
            return false
        }
    }
    
    func deleteAccount(_ account: MT5Account) async {
        do {
            try await accountService.deleteAccount(id: account.id)
        } catch {
            // Error handled in service
        }
    }
    
    func setPrimary(_ account: MT5Account) async {
        do {
            try await accountService.setPrimaryAccount(id: account.id)
        } catch {
            // Error handled in service
        }
    }
}
