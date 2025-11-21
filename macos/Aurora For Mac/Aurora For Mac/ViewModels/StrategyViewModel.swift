import Foundation

@MainActor
class StrategyViewModel: ObservableObject {
    @Published var strategies: [Strategy] = []
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    private let apiService = APIService.shared

    func loadStrategies() async {
        isLoading = true
        errorMessage = nil

        do {
            let strategiesDict = try await apiService.fetchStrategies()
            strategies = Array(strategiesDict.values).sorted { $0.name < $1.name }
        } catch {
            errorMessage = "Failed to load strategies: \(error.localizedDescription)"
        }

        isLoading = false
    }

    func uploadStrategy(_ strategy: Strategy) async -> Bool {
        isLoading = true
        errorMessage = nil

        do {
            try await apiService.uploadStrategy(strategy)
            await loadStrategies()  // Refresh list
            return true
        } catch {
            errorMessage = "Failed to upload strategy: \(error.localizedDescription)"
            isLoading = false
            return false
        }
    }

    func deleteStrategy(_ strategy: Strategy) {
        strategies.removeAll { $0.id == strategy.id }
    }
}
