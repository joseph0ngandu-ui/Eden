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

            Divider()

            // Content
            if viewModel.isLoading {
                ProgressView("Loading strategies...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.strategies.isEmpty {
                ContentUnavailableView {
                    Label("No Strategies", systemImage: "chart.line.uptrend.xyaxis")
                } description: {
                    Text("Create your first strategy or train one using ML")
                }
            } else {
                List(selection: $selectedStrategy) {
                    ForEach(viewModel.strategies) { strategy in
                        StrategyRow(strategy: strategy)
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

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(strategy.name)
                    .font(.headline)

                if let description = strategy.description {
                    Text(description)
                        .font(.caption)
                        .foregroundColor(.secondary)
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

            VStack(alignment: .trailing, spacing: 4) {
                if strategy.isActive {
                    Label("Active", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                }

                if let mode = strategy.mode {
                    Text(mode)
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            mode == "LIVE" ? Color.green.opacity(0.2) : Color.orange.opacity(0.2)
                        )
                        .cornerRadius(4)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    StrategyListView()
        .environmentObject(StrategyViewModel())
        .frame(width: 800, height: 600)
}

