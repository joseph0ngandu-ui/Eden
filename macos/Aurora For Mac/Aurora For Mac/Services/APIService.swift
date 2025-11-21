import Foundation

class APIService: ObservableObject {
    static let shared = APIService()

    @Published var baseURL: String = "https://edenbot.duckdns.org:8443"

    private var authToken: String? {
        get { UserDefaults.standard.string(forKey: "authToken") }
        set { UserDefaults.standard.set(newValue, forKey: "authToken") }
    }

    private init() {}

    // MARK: - Authentication

    func login(email: String, password: String) async throws -> String {
        let url = URL(string: "\(baseURL)/auth/login-local")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = ["email": email, "password": password]
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw APIError.authenticationFailed
        }

        let tokenResponse = try JSONDecoder().decode(TokenResponse.self, from: data)
        authToken = tokenResponse.accessToken
        return tokenResponse.accessToken
    }

    // MARK: - Strategies

    func fetchStrategies() async throws -> [String: Strategy] {
        let url = URL(string: "\(baseURL)/strategies")!
        var request = URLRequest(url: url)
        request.setValue("Bearer \(authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw APIError.networkError
        }

        return try JSONDecoder().decode([String: Strategy].self, from: data)
    }

    func uploadStrategy(_ strategy: Strategy) async throws {
        let url = URL(string: "\(baseURL)/strategies")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(authToken ?? "")", forHTTPHeaderField: "Authorization")

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        request.httpBody = try encoder.encode(strategy)

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw APIError.uploadFailed
        }
    }

    // MARK: - Supporting Types

    struct TokenResponse: Codable {
        let accessToken: String
        let tokenType: String

        enum CodingKeys: String, CodingKey {
            case accessToken = "access_token"
            case tokenType = "token_type"
        }
    }

    enum APIError: Error {
        case authenticationFailed
        case networkError
        case uploadFailed
        case invalidResponse
    }
}
