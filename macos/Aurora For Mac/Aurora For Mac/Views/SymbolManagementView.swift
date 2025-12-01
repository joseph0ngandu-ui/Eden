import SwiftUI

struct SymbolManagementView: View {
    @StateObject private var strategyService = StrategyService.shared
    @State private var availableSymbols: [String] = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "ETHUSD"] // Default list, ideally fetched
    @State private var enabledSymbols: Set<String> = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var searchText = ""
    
    var filteredSymbols: [String] {
        if searchText.isEmpty {
            return availableSymbols
        } else {
            return availableSymbols.filter { $0.localizedCaseInsensitiveContains(searchText) }
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Symbol Management")
                    .font(.title)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button {
                    Task {
                        await loadData()
                    }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
            .padding()
            
            Divider()
            
            // Search Bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search symbols...", text: $searchText)
                    .textFieldStyle(.plain)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            
            Divider()
            
            // Content
            if isLoading {
                ProgressView("Loading configuration...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(filteredSymbols, id: \.self) { symbol in
                        HStack {
                            Text(symbol)
                                .font(.headline)
                            
                            Spacer()
                            
                            Toggle("", isOn: Binding(
                                get: { enabledSymbols.contains(symbol) },
                                set: { isEnabled in
                                    if isEnabled {
                                        enabledSymbols.insert(symbol)
                                    } else {
                                        enabledSymbols.remove(symbol)
                                    }
                                    saveChanges()
                                }
                            ))
                            .toggleStyle(.switch)
                        }
                        .padding(.vertical, 4)
                    }
                }
                .listStyle(.inset)
            }
            
            if let error = errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                    Text(error)
                    Spacer()
                    Button("Dismiss") {
                        errorMessage = nil
                    }
                }
                .padding()
                .background(Color.red.opacity(0.1))
            }
        }
        .task {
            await loadData()
        }
    }
    
    private func loadData() async {
        isLoading = true
        errorMessage = nil
        
        do {
            // 1. Get current config to see enabled symbols
            let config = try await strategyService.getStrategyConfig()
            enabledSymbols = Set(config.enabledSymbols)
            
            // 2. Get all available symbols (if there's an endpoint, otherwise use hardcoded defaults + enabled)
            // For now, we'll merge hardcoded with enabled to ensure we show everything
            let fetchedSymbols = try await strategyService.getTradableSymbols()
            if !fetchedSymbols.isEmpty {
                 // If backend returns list of ALL symbols, use it. 
                 // If it returns only enabled, we might need another endpoint or just use defaults.
                 // Assuming getTradableSymbols returns ALL available symbols from broker.
                 availableSymbols = fetchedSymbols
            } else {
                 // Fallback: merge defaults with enabled
                 var all = Set(availableSymbols)
                 all.formUnion(enabledSymbols)
                 availableSymbols = Array(all).sorted()
            }
            
            isLoading = false
        } catch {
            isLoading = false
            errorMessage = "Failed to load data: \(error.localizedDescription)"
        }
    }
    
    private func saveChanges() {
        Task {
            do {
                var config = try await strategyService.getStrategyConfig()
                config.enabledSymbols = Array(enabledSymbols).sorted()
                try await strategyService.updateStrategyConfig(config)
            } catch {
                errorMessage = "Failed to save changes: \(error.localizedDescription)"
            }
        }
    }
}

#Preview {
    SymbolManagementView()
}
