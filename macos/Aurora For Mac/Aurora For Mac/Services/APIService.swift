import Foundation

class APIService: ObservableObject {
    static let shared = APIService()

    @Published var baseURL: String = "https://edenbot.duckdns.org:8443"

    private var authToken: String? {
        get { UserDefaults.standard.string(forKey: "authToken") }
        set { UserDefaults.standard.set(newValue, forKey: "authToken") }
    }

    private init() {}

    // MARK: - ML Endpoints

    func startMLTraining(
        symbol: String, modelType: String, dataLength: Int, epochs: Int = 50,
        learningRate: Double = 0.001
    ) async throws {
        let endpoint = "/ml/train"
        let body = TrainingRequest(
            symbol: symbol,
            model_type: modelType.lowercased(),
            data_length: dataLength,
            epochs: epochs,
            batch_size: 32,
            learning_rate: learningRate
        )

        guard let url = URL(string: baseURL + endpoint) else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        request.httpBody = try JSONEncoder().encode(body)

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            (200...299).contains(httpResponse.statusCode)
        else {
            throw APIError.requestFailed
        }
    }

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
        case invalidURL
        case requestFailed
        case decodingFailed
    }
}
