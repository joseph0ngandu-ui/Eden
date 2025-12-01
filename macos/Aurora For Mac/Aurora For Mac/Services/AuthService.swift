import Combine
import Foundation
import Security

@MainActor
class AuthService: ObservableObject {
    static let shared = AuthService()

    @Published var isAuthenticated: Bool = false
    @Published var currentUser: String?

    private let apiService = APIService.shared

    private init() {
        checkAuthStatus()
    }

    func login(email: String, password: String) async throws {
        let token = try await apiService.login(email: email, password: password)

        // Save credentials to Keychain
        try saveToKeychain(email: email, password: password)

        await MainActor.run {
            self.isAuthenticated = true
            self.currentUser = email

            // Connect WebSockets
            WebSocketService.shared.connectAll()
        }
    }

    func logout() {
        deleteFromKeychain()
        isAuthenticated = false
        currentUser = nil

        // Disconnect WebSockets
        WebSocketService.shared.disconnectAll()
    }

    private func checkAuthStatus() {
        if loadFromKeychain() != nil {
            isAuthenticated = true
            // Connect WebSockets if already logged in
            WebSocketService.shared.connectAll()
        }
    }

    // MARK: - Keychain Helper

    private func saveToKeychain(email: String, password: String) throws {
        let data = password.data(using: .utf8)!

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: email,
            kSecValueData as String: data,
        ]

        SecItemDelete(query as CFDictionary)
        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw KeychainError.saveFailed
        }
    }

    private func loadFromKeychain() -> (email: String, password: String)? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecReturnAttributes as String: true,
            kSecReturnData as String: true,
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)

        guard status == errSecSuccess,
            let existingItem = item as? [String: Any],
            let account = existingItem[kSecAttrAccount as String] as? String,
            let passwordData = existingItem[kSecValueData as String] as? Data,
            let password = String(data: passwordData, encoding: .utf8)
        else {
            return nil
        }

        return (account, password)
    }

    private func deleteFromKeychain() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword
        ]
        SecItemDelete(query as CFDictionary)
    }

    enum KeychainError: Error {
        case saveFailed
        case loadFailed
    }
}
