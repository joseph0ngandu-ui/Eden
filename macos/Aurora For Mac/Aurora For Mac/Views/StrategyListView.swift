import SwiftUI

struct StrategyListView: View {
    @EnvironmentObject var viewModel: StrategyViewModel
    @State private var selectedStrategy: Strategy?
    @State private var showingEditor = false
    @State private var editingStrategy: Strategy?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Strategies")
                    .font(.title)
                    .fontWeight(.bold)

                Spacer()

                Button {
                    editingStrategy = Strategy(name: "New Strategy")
                    showingEditor = true
                } label: {
                    Label("New Strategy", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)

                Button {
                    Task {
                        await viewModel.loadStrategies()
                    }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
            .padding()

            // Filter Tabs
            Picker("Filter", selection: $viewModel.filterMode) {
                ForEach(StrategyViewModel.FilterMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            Divider()

            // Content
            if viewModel.isLoading {
                ProgressView("Loading strategies...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.filteredStrategies.isEmpty {
                ContentUnavailableView {
                    Label("No Strategies", systemImage: "chart.line.uptrend.xyaxis")
                } description: {
                    Text("Create your first strategy or train one using ML")
                }
            } else {
                List(selection: $selectedStrategy) {
                    ForEach(viewModel.filteredStrategies) { strategy in
                        StrategyRow(strategy: strategy, viewModel: viewModel)
                            .tag(strategy)
                            .contextMenu {
                                Button("Edit") {
                                    editingStrategy = strategy
                                    showingEditor = true
                                }

                                Button("Duplicate") {
                                    let duplicate = strategy.duplicated()
                                    editingStrategy = duplicate
                                    showingEditor = true
                                }
                                
                                Divider()
                                
                                if strategy.isActive {
                                    Button("Deactivate") {
                                        Task {
                                            await viewModel.deactivateStrategy(strategy)
                                        }
                                    }
                                } else {
                                    Button("Activate") {
                                        Task {
                                            await viewModel.activateStrategy(strategy)
                                        }
                                    }
                                }
                                
                                if strategy.mode == "PAPER" && strategy.validated == true {
                                    Button("Promote to LIVE") {
                                        Task {
                                            await viewModel.promoteStrategy(strategy)
                                        }
                                    }
                                }

                                Divider()

                                Button("Delete", role: .destructive) {
                                    viewModel.deleteStrategy(strategy)
                                }
                            }
                    }
                }
                .listStyle(.inset)
            }

            if let error = viewModel.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                    Text(error)
                    Spacer()
                    Button("Dismiss") {
                        viewModel.errorMessage = nil
                    }
                }
                .padding()
                .background(Color.red.opacity(0.1))
            }
        }
        .sheet(isPresented: $showingEditor) {
            if let strategy = editingStrategy {
                StrategyEditorView(
                    strategy: strategy,
                    onSave: { savedStrategy in
                        Task {
                            let success = await viewModel.uploadStrategy(savedStrategy)
                            if success {
                                showingEditor = false
                            }
                        }
                    })
            }
        }
        .task {
            await viewModel.loadStrategies()
        }
    }
}

struct StrategyRow: View {
    let strategy: Strategy
    let viewModel: StrategyViewModel
    @State private var showPromoteConfirmation = false

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(strategy.name)
                        .font(.headline)
                    
                    // Status Badge
                    if let mode = strategy.mode {
                        Text(mode)
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                mode == "LIVE" ? Color.green.opacity(0.2) : Color.orange.opacity(0.2)
                            )
                            .foregroundColor(mode == "LIVE" ? .green : .orange)
                            .cornerRadius(4)
                    }
                    
                    // Validated Badge
                    if strategy.validated == true {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundColor(.blue)
                            .font(.caption)
                    }
                }

                if let description = strategy.description {
                    Text(description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }

                HStack(spacing: 12) {
                    Label(strategy.parameters.timeframe, systemImage: "clock")
                    Label("\(strategy.parameters.maxPositions) pos", systemImage: "chart.bar")
                    Label(
                        "\(String(format: "%.1f%%", strategy.parameters.riskPerTrade))",
                        systemImage: "percent")
                }
                .font(.caption2)
                .foregroundColor(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 8) {
                // Active Status
                if strategy.isActive {
                    Label("Active", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                }
                
                // Action Buttons
                HStack(spacing: 8) {
                    // Activate/Deactivate Button
                    Button {
                        Task {
                            if strategy.isActive {
                                await viewModel.deactivateStrategy(strategy)
                            } else {
                                await viewModel.activateStrategy(strategy)
                            }
                        }
                    } label: {
                        Image(systemImage: strategy.isActive ? "pause.circle" : "play.circle")
                            .foregroundColor(strategy.isActive ? .orange : .green)
                    }
                    .buttonStyle(.plain)
                    .help(strategy.isActive ? "Deactivate" : "Activate")
                    
                    // Promote to LIVE Button
                    if strategy.mode == "PAPER" && strategy.validated == true {
                        Button {
                            showPromoteConfirmation = true
                        } label: {
                            Image(systemImage: "arrow.up.circle.fill")
                                .foregroundColor(.blue)
                        }
                        .buttonStyle(.plain)
                        .help("Promote to LIVE")
                    }
                }
            }
        }
        .padding(.vertical, 4)
        .alert("Promote to LIVE?", isPresented: $showPromoteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Promote", role: .destructive) {
                Task {
                    await viewModel.promoteStrategy(strategy)
                }
            }
        } message: {
            Text("This will promote '\(strategy.name)' from PAPER to LIVE trading mode. Real money will be at risk. Are you sure?")
        }
    }
}

#Preview {
    StrategyListView()
        .environmentObject(StrategyViewModel())
        .frame(width: 800, height: 600)
}

