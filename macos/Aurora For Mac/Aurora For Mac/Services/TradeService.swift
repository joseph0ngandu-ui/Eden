import Combine
import Foundation

class TradeService: ObservableObject {
    static let shared = TradeService()

    @Published var openPositions: [Position] = []
    @Published var tradeHistory: [Trade] = []
    @Published var recentTrades: [Trade] = []
    @Published var isLoading: Bool = false
    @Published var error: String?

    private let apiService = APIService.shared
    private var cancellables = Set<AnyCancellable>()

    private init() {}

    // MARK: - Positions

    func getOpenPositions() async throws -> [Position] {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/trades/open")!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TradeServiceError.fetchFailed
        }

        let positions = try JSONDecoder().decode([Position].self, from: data)

        await MainActor.run {
            self.openPositions = positions
        }

        return positions
    }

    // MARK: - Trade History

    func getTradeHistory(limit: Int? = nil, offset: Int? = nil) async throws -> [Trade] {
        isLoading = true
        defer { isLoading = false }

        var urlString = "\(apiService.baseURL)/trades/history"
        var queryItems: [String] = []

        if let limit = limit {
            queryItems.append("limit=\(limit)")
        }
        if let offset = offset {
            queryItems.append("offset=\(offset)")
        }

        if !queryItems.isEmpty {
            urlString += "?" + queryItems.joined(separator: "&")
        }

        let url = URL(string: urlString)!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TradeServiceError.fetchFailed
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let trades = try decoder.decode([Trade].self, from: data)

        await MainActor.run {
            self.tradeHistory = trades
        }

        return trades
    }

    func getRecentTrades(limit: Int = 10) async throws -> [Trade] {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/trades/recent?limit=\(limit)")!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TradeServiceError.fetchFailed
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let trades = try decoder.decode([Trade].self, from: data)

        await MainActor.run {
            self.recentTrades = trades
        }

        return trades
    }

    // MARK: - Close Trade

    func closeTrade(positionId: String) async throws {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/trades/close")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let body = ["position_id": positionId]
        request.httpBody = try JSONEncoder().encode(body)

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            (200...299).contains(httpResponse.statusCode)
        else {
            throw TradeServiceError.closeFailed
        }

        // Refresh positions after closing
        try await getOpenPositions()
    }

    // MARK: - Trade Logs

    func getTradeLogs(limit: Int = 100) async throws -> String {
        let url = URL(string: "\(apiService.baseURL)/trades/logs?limit=\(limit)")!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TradeServiceError.fetchFailed
        }

        return String(data: data, encoding: .utf8) ?? ""
    }

    // MARK: - Auto-refresh

    func startAutoRefresh(interval: TimeInterval = 3.0) {
        Timer.publish(every: interval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    try? await self?.getOpenPositions()
                    try? await self?.getRecentTrades(limit: 10)
                }
            }
            .store(in: &cancellables)
    }

    func stopAutoRefresh() {
        cancellables.removeAll()
    }

    // MARK: - Error Types

    enum TradeServiceError: Error, LocalizedError {
        case fetchFailed
        case closeFailed
        case networkError

        var errorDescription: String? {
            switch self {
            case .fetchFailed:
                return "Failed to fetch trade data"
            case .closeFailed:
                return "Failed to close trade"
            case .networkError:
                return "Network connection error"
            }
        }
    }
}
