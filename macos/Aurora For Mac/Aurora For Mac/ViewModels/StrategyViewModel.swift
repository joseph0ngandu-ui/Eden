import Combine
import Foundation
import SwiftUI

class StrategyViewModel: ObservableObject {
    @Published var strategies: [Strategy] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var filterMode: FilterMode = .all
    
    private let strategyService = StrategyService.shared
    private var cancellables = Set<AnyCancellable>()
    
    enum FilterMode: String, CaseIterable {
        case all = "All"
        case active = "Active"
        case validated = "Validated"
        case paper = "Paper"
        case live = "Live"
    }
    
    init() {
        // Subscribe to strategy service updates
        strategyService.$strategies
            .receive(on: DispatchQueue.main)
            .sink { [weak self] strategies in
                self?.strategies = strategies
            }
            .store(in: &cancellables)
        
        strategyService.$isLoading
            .receive(on: DispatchQueue.main)
            .assign(to: &$isLoading)
        
        strategyService.$errorMessage
            .receive(on: DispatchQueue.main)
            .assign(to: &$errorMessage)
    }
    
    // MARK: - Computed Properties
    
    var filteredStrategies: [Strategy] {
        switch filterMode {
        case .all:
            return strategies
        case .active:
            return strategies.filter { $0.isActive }
        case .validated:
            return strategies.filter { $0.validated == true }
        case .paper:
            return strategies.filter { $0.mode == "PAPER" }
        case .live:
            return strategies.filter { $0.mode == "LIVE" }
        }
    }
    
    // MARK: - Actions
    
    func loadStrategies() async {
        do {
            try await strategyService.fetchStrategies()
        } catch {
            await MainActor.run {
                errorMessage = "Failed to load strategies: \(error.localizedDescription)"
            }
        }
    }
    
    func uploadStrategy(_ strategy: Strategy) async -> Bool {
        do {
            // Convert strategy to JSON and upload
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(strategy)
            
            _ = try await APIService.shared.performRequest(
                endpoint: "/strategies",
                method: "POST",
                body: data
            )
            
            try await strategyService.fetchStrategies()
            return true
        } catch {
            await MainActor.run {
                errorMessage = "Failed to upload strategy: \(error.localizedDescription)"
            }
            return false
        }
    }
    
    func activateStrategy(_ strategy: Strategy) async {
        do {
            try await strategyService.activateStrategy(id: strategy.id)
        } catch {
            await MainActor.run {
                errorMessage = "Failed to activate strategy: \(error.localizedDescription)"
            }
        }
    }
    
    func deactivateStrategy(_ strategy: Strategy) async {
        do {
            try await strategyService.deactivateStrategy(id: strategy.id)
        } catch {
            await MainActor.run {
                errorMessage = "Failed to deactivate strategy: \(error.localizedDescription)"
            }
        }
    }
    
    func promoteStrategy(_ strategy: Strategy) async {
        do {
            try await strategyService.promoteStrategy(id: strategy.id)
        } catch {
            await MainActor.run {
                errorMessage = "Failed to promote strategy: \(error.localizedDescription)"
            }
        }
    }
    
    func updatePolicy(_ strategy: Strategy, policy: StrategyPolicy) async {
        do {
            try await strategyService.updatePolicy(id: strategy.id, policy: policy)
        } catch {
            await MainActor.run {
                errorMessage = "Failed to update policy: \(error.localizedDescription)"
            }
        }
    }
    
    func deleteStrategy(_ strategy: Strategy) {
        // For now, just remove from local list
        // In a real implementation, this would call a delete endpoint
        strategies.removeAll { $0.id == strategy.id }
    }
}
