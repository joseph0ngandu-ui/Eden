import Combine
import Foundation

class StrategyService: ObservableObject {
    static let shared = StrategyService()

    @Published var strategies: [Strategy] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let apiService = APIService.shared

    private init() {}

    // MARK: - Fetch Strategies

    func fetchStrategies() async throws {
        await MainActor.run { isLoading = true }

        do {
            let data = try await apiService.performRequest(endpoint: "/strategies", method: "GET")

            // Backend returns a dictionary [ID: Strategy]
            let strategyDict = try JSONDecoder.iso8601.decode([String: Strategy].self, from: data)
            let strategyList = Array(strategyDict.values).sorted { $0.displayName < $1.displayName }

            await MainActor.run {
                self.strategies = strategyList
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

    // MARK: - Strategy Actions

    func uploadStrategy(fileURL: URL) async throws {
        // Read file content
        guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else {
            throw NSError(
                domain: "StrategyService", code: 400,
                userInfo: [NSLocalizedDescriptionKey: "Failed to read file"])
        }

        // Parse python file to extract metadata (simplified for now, just sending raw content if backend supported it,
        // but backend expects JSON config. Assuming we are uploading a JSON config or constructing one)

        // For now, let's assume we are uploading a JSON configuration file
        let data = try Data(contentsOf: fileURL)
        let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        guard var strategyData = jsonObject else {
            throw NSError(
                domain: "StrategyService", code: 400,
                userInfo: [NSLocalizedDescriptionKey: "Invalid JSON file"])
        }

        // Ensure ID exists
        if strategyData["id"] == nil {
            strategyData["id"] = UUID().uuidString
        }

        let jsonData = try JSONSerialization.data(withJSONObject: strategyData)

        _ = try await apiService.performRequest(
            endpoint: "/strategies", method: "POST", body: jsonData)
        try await fetchStrategies()
    }

    func activateStrategy(id: String) async throws {
        _ = try await apiService.performRequest(
            endpoint: "/strategies/\(id)/activate", method: "PUT")
        try await fetchStrategies()
    }

    func deactivateStrategy(id: String) async throws {
        _ = try await apiService.performRequest(
            endpoint: "/strategies/\(id)/deactivate", method: "PUT")
        try await fetchStrategies()
    }

    func promoteStrategy(id: String) async throws {
        _ = try await apiService.performRequest(
            endpoint: "/strategies/\(id)/promote", method: "PUT")
        try await fetchStrategies()
    }

    func updatePolicy(id: String, policy: StrategyPolicy) async throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(policy)
        _ = try await apiService.performRequest(
            endpoint: "/strategies/\(id)/policy", method: "PATCH", body: data)
        try await fetchStrategies()
    }
    
    // MARK: - Filtered Strategies
    
    func getActiveStrategies() async throws -> [Strategy] {
        let data = try await apiService.performRequest(endpoint: "/strategies/active", method: "GET")
        let strategyDict = try JSONDecoder.iso8601.decode([String: Strategy].self, from: data)
        return Array(strategyDict.values).sorted { $0.displayName < $1.displayName }
    }
    
    func getValidatedStrategies() async throws -> [Strategy] {
        let data = try await apiService.performRequest(endpoint: "/strategies/validated", method: "GET")
        let strategyDict = try JSONDecoder.iso8601.decode([String: Strategy].self, from: data)
        return Array(strategyDict.values).sorted { $0.displayName < $1.displayName }
    }
    
    // MARK: - Configuration Management
    
    func getStrategyConfig() async throws -> StrategyConfig {
        let data = try await apiService.performRequest(endpoint: "/strategy/config", method: "GET")
        return try JSONDecoder.iso8601.decode(StrategyConfig.self, from: data)
    }
    
    func updateStrategyConfig(_ config: StrategyConfig) async throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(config)
        _ = try await apiService.performRequest(
            endpoint: "/strategy/config", method: "POST", body: data)
    }
    
    func getTradableSymbols() async throws -> [String] {
        let data = try await apiService.performRequest(endpoint: "/strategy/symbols", method: "GET")
        let response = try JSONDecoder().decode([String: [String]].self, from: data)
        return response["symbols"] ?? []
    }
}


// Extension for ISO8601 decoding
extension JSONDecoder {
    static var iso8601: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
