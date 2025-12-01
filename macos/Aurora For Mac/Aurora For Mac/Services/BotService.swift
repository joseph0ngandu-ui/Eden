import Combine
import Foundation

class BotService: ObservableObject {
    static let shared = BotService()

    @Published var botStatus: BotStatus?
    @Published var isLoading: Bool = false
    @Published var error: String?

    private let apiService = APIService.shared
    private var cancellables = Set<AnyCancellable>()

    private init() {}

    // MARK: - Bot Status

    func getBotStatus() async throws -> BotStatus {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/bot/status")!
        var request = URLRequest(url: url)
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw BotServiceError.statusFetchFailed
        }

        let status = try JSONDecoder().decode(BotStatus.self, from: data)

        await MainActor.run {
            self.botStatus = status
        }

        return status
    }

    // MARK: - Bot Control

    func startBot() async throws {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/bot/start")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            (200...299).contains(httpResponse.statusCode)
        else {
            throw BotServiceError.startFailed
        }

        // Refresh status after starting
        try await getBotStatus()
    }

    func stopBot() async throws {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/bot/stop")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            (200...299).contains(httpResponse.statusCode)
        else {
            throw BotServiceError.stopFailed
        }

        // Refresh status after stopping
        try await getBotStatus()
    }

    func pauseBot() async throws {
        isLoading = true
        defer { isLoading = false }

        let url = URL(string: "\(apiService.baseURL)/bot/pause")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(
            "Bearer \(apiService.authToken ?? "")", forHTTPHeaderField: "Authorization")

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            (200...299).contains(httpResponse.statusCode)
        else {
            throw BotServiceError.pauseFailed
        }

        // Refresh status after pausing
        try await getBotStatus()
    }

    // MARK: - Auto-refresh

    func startAutoRefresh(interval: TimeInterval = 5.0) {
        Timer.publish(every: interval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    try? await self?.getBotStatus()
                }
            }
            .store(in: &cancellables)
    }

    func stopAutoRefresh() {
        cancellables.removeAll()
    }

    // MARK: - Error Types

    enum BotServiceError: Error, LocalizedError {
        case statusFetchFailed
        case startFailed
        case stopFailed
        case pauseFailed
        case networkError

        var errorDescription: String? {
            switch self {
            case .statusFetchFailed:
                return "Failed to fetch bot status"
            case .startFailed:
                return "Failed to start bot"
            case .stopFailed:
                return "Failed to stop bot"
            case .pauseFailed:
                return "Failed to pause bot"
            case .networkError:
                return "Network connection error"
            }
        }
    }
}
