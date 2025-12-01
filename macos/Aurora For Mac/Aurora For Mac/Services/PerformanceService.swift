import Combine
import Foundation

class PerformanceService: ObservableObject {
    static let shared = PerformanceService()

    @Published var stats: PerformanceStats?
    @Published var equityCurve: [EquityPoint] = []
    @Published var dailySummaries: [DailySummary] = []
    @Published var isLoading: Bool = false
    @Published var error: String?

    private let apiService = APIService.shared

    private init() {}

    // MARK: - Performance Stats

    func getStats() async throws -> PerformanceStats {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/performance/stats")!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw PerformanceServiceError.fetchFailed
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let performanceStats = try decoder.decode(PerformanceStats.self, from: data)

        await MainActor.run {
            self.stats = performanceStats
        }

        return performanceStats
    }

    // MARK: - Equity Curve

    func getEquityCurve(days: Int? = nil) async throws -> [EquityPoint] {
        isLoading = true
        defer { isLoading = false }

        var urlString = "\(apiService.baseURL)/performance/equity-curve"
        if let days = days {
            urlString += "?days=\(days)"
        }

        let url = URL(string: urlString)!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw PerformanceServiceError.fetchFailed
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        // Handle both array and object response
        if let points = try? decoder.decode([EquityPoint].self, from: data) {
            await MainActor.run {
                self.equityCurve = points
            }
            return points
        } else if let json = try? JSONDecoder().decode([String: [[String: Any]]].self, from: data),
            let dataPoints = json["equity_curve"]
        {
            // Handle nested response format if needed
            let points = try parseEquityPoints(from: dataPoints)
            await MainActor.run {
                self.equityCurve = points
            }
            return points
        }

        throw PerformanceServiceError.decodingFailed
    }

    private func parseEquityPoints(from data: [[String: Any]]) throws -> [EquityPoint] {
        var points: [EquityPoint] = []
        let dateFormatter = ISO8601DateFormatter()

        for (index, dict) in data.enumerated() {
            let timestamp: Date
            if let timestampString = dict["timestamp"] as? String {
                timestamp = dateFormatter.date(from: timestampString) ?? Date()
            } else {
                timestamp = Date()
            }

            let point = EquityPoint(
                id: dict["id"] as? String ?? "point_\(index)",
                timestamp: timestamp,
                balance: dict["balance"] as? Double ?? 0,
                equity: dict["equity"] as? Double ?? 0,
                profit: dict["profit"] as? Double ?? 0
            )
            points.append(point)
        }

        return points
    }

    // MARK: - Daily Summary

    func getDailySummary(days: Int = 30) async throws -> [DailySummary] {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/performance/daily-summary?days=\(days)")!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw PerformanceServiceError.fetchFailed
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let summaries = try decoder.decode([DailySummary].self, from: data)

        await MainActor.run {
            self.dailySummaries = summaries
        }

        return summaries
    }

    // MARK: - Refresh All

    func refreshAllData() async throws {
        async let stats = getStats()
        async let equity = getEquityCurve()
        async let daily = getDailySummary()

        _ = try await (stats, equity, daily)
    }

    // MARK: - Error Types

    enum PerformanceServiceError: Error, LocalizedError {
        case fetchFailed
        case decodingFailed
        case networkError

        var errorDescription: String? {
            switch self {
            case .fetchFailed:
                return "Failed to fetch performance data"
            case .decodingFailed:
                return "Failed to decode performance data"
            case .networkError:
                return "Network connection error"
            }
        }
    }
}
