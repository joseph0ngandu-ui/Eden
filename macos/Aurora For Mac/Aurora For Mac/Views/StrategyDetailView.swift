import SwiftUI

struct StrategyDetailView: View {
    let strategy: Strategy
    @ObservedObject var viewModel: StrategyViewModel
    @Environment(\.dismiss) var dismiss
    
    @State private var showPromoteConfirmation = false
    @State private var showDeactivateConfirmation = false
    @State private var showDeleteConfirmation = false
    @State private var editingPolicy = false
    @State private var policy: StrategyPolicy
    
    init(strategy: Strategy, viewModel: StrategyViewModel) {
        self.strategy = strategy
        self.viewModel = viewModel
        _policy = State(initialValue: strategy.policy ?? StrategyPolicy())
    }
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // MARK: - Overview Section
                    GroupBox("Overview") {
                        VStack(alignment: .leading, spacing: 12) {
                            DetailRow(label: "Name", value: strategy.name)
                            
                            if let description = strategy.description {
                                DetailRow(label: "Description", value: description)
                            }
                            
                            HStack {
                                Text("Status:")
                                    .foregroundColor(.secondary)
                                Spacer()
                                HStack(spacing: 8) {
                                    if strategy.isActive {
                                        Label("Active", systemImage: "checkmark.circle.fill")
                                            .foregroundColor(.green)
                                    } else {
                                        Label("Inactive", systemImage: "circle")
                                            .foregroundColor(.gray)
                                    }
                                    
                                    if let mode = strategy.mode {
                                        Text(mode)
                                            .font(.caption)
                                            .fontWeight(.semibold)
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 4)
                                            .background(
                                                mode == "LIVE" ? Color.green.opacity(0.2) : Color.orange.opacity(0.2)
                                            )
                                            .foregroundColor(mode == "LIVE" ? .green : .orange)
                                            .cornerRadius(4)
                                    }
                                    
                                    if strategy.validated == true {
                                        Image(systemName: "checkmark.seal.fill")
                                            .foregroundColor(.blue)
                                    }
                                }
                                .font(.caption)
                            }
                            
                            if let createdAt = strategy.createdAt {
                                DetailRow(label: "Created", value: createdAt.formatted(date: .abbreviated, time: .shortened))
                            }
                            
                            if let updatedAt = strategy.updatedAt {
                                DetailRow(label: "Updated", value: updatedAt.formatted(date: .abbreviated, time: .shortened))
                            }
                        }
                    }
                    
                    // MARK: - Parameters Section
                    GroupBox("Parameters") {
                        VStack(alignment: .leading, spacing: 12) {
                            DetailRow(label: "Timeframe", value: strategy.parameters.timeframe)
                            DetailRow(label: "Max Positions", value: "\(strategy.parameters.maxPositions)")
                            DetailRow(label: "Risk per Trade", value: String(format: "%.1f%%", strategy.parameters.riskPerTrade))
                            DetailRow(label: "Stop Loss", value: String(format: "%.1f%%", strategy.parameters.stopLossPercent))
                            DetailRow(label: "Take Profit", value: String(format: "%.1f%%", strategy.parameters.takeProfitPercent))
                        }
                    }
                    
                    // MARK: - Indicators Section
                    if !strategy.indicators.isEmpty {
                        GroupBox("Indicators") {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(strategy.indicators, id: \.self) { indicator in
                                    HStack {
                                        Image(systemName: "chart.line.uptrend.xyaxis")
                                            .foregroundColor(.blue)
                                        Text(indicator)
                                    }
                                    .font(.caption)
                                }
                            }
                        }
                    }
                    
                    // MARK: - Policy Section
                    GroupBox("Policy Settings") {
                        VStack(alignment: .leading, spacing: 12) {
                            if editingPolicy {
                                // Editable policy fields
                                HStack {
                                    Text("Max Positions:")
                                    Spacer()
                                    TextField("", value: Binding(
                                        get: { policy.maxPositions ?? strategy.parameters.maxPositions },
                                        set: { policy.maxPositions = $0 }
                                    ), format: .number)
                                    .frame(width: 80)
                                    .textFieldStyle(.roundedBorder)
                                }
                                
                                HStack {
                                    Text("Risk per Trade:")
                                    Spacer()
                                    TextField("", value: Binding(
                                        get: { policy.riskPerTrade ?? strategy.parameters.riskPerTrade },
                                        set: { policy.riskPerTrade = $0 }
                                    ), format: .number)
                                    .frame(width: 80)
                                    .textFieldStyle(.roundedBorder)
                                    Text("%")
                                }
                                
                                HStack {
                                    Button("Cancel") {
                                        editingPolicy = false
                                        policy = strategy.policy ?? StrategyPolicy()
                                    }
                                    .buttonStyle(.bordered)
                                    
                                    Button("Save Policy") {
                                        Task {
                                            await viewModel.updatePolicy(strategy, policy: policy)
                                            editingPolicy = false
                                        }
                                    }
                                    .buttonStyle(.borderedProminent)
                                }
                            } else {
                                // Display current policy
                                if let maxPos = policy.maxPositions {
                                    DetailRow(label: "Max Positions Override", value: "\(maxPos)")
                                }
                                if let risk = policy.riskPerTrade {
                                    DetailRow(label: "Risk Override", value: String(format: "%.1f%%", risk))
                                }
                                if policy.maxPositions == nil && policy.riskPerTrade == nil {
                                    Text("Using default parameters")
                                        .foregroundColor(.secondary)
                                        .font(.caption)
                                }
                                
                                Button("Edit Policy") {
                                    editingPolicy = true
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                    
                    // MARK: - Controls Section
                    GroupBox("Actions") {
                        VStack(spacing: 12) {
                            // Activate/Deactivate
                            if strategy.isActive {
                                Button {
                                    showDeactivateConfirmation = true
                                } label: {
                                    Label("Deactivate Strategy", systemImage: "pause.circle.fill")
                                        .frame(maxWidth: .infinity)
                                }
                                .buttonStyle(.bordered)
                                .tint(.orange)
                            } else {
                                Button {
                                    Task {
                                        await viewModel.activateStrategy(strategy)
                                    }
                                } label: {
                                    Label("Activate Strategy", systemImage: "play.circle.fill")
                                        .frame(maxWidth: .infinity)
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.green)
                            }
                            
                            // Promote to LIVE
                            if strategy.mode == "PAPER" && strategy.validated == true {
                                Button {
                                    showPromoteConfirmation = true
                                } label: {
                                    Label("Promote to LIVE", systemImage: "arrow.up.circle.fill")
                                        .frame(maxWidth: .infinity)
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.blue)
                            }
                            
                            Divider()
                            
                            // Delete
                            Button(role: .destructive) {
                                showDeleteConfirmation = true
                            } label: {
                                Label("Delete Strategy", systemImage: "trash")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .disabled(strategy.isActive)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle(strategy.name)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
        .alert("Promote to LIVE?", isPresented: $showPromoteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Promote", role: .destructive) {
                Task {
                    await viewModel.promoteStrategy(strategy)
                    dismiss()
                }
            }
        } message: {
            Text("This will promote '\(strategy.name)' from PAPER to LIVE trading mode. Real money will be at risk. Are you sure?")
        }
        .alert("Deactivate Strategy?", isPresented: $showDeactivateConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Deactivate") {
                Task {
                    await viewModel.deactivateStrategy(strategy)
                    dismiss()
                }
            }
        } message: {
            Text("This will deactivate '\(strategy.name)'. Any open positions will remain open.")
        }
        .alert("Delete Strategy?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                viewModel.deleteStrategy(strategy)
                dismiss()
            }
        } message: {
            Text("This will permanently delete '\(strategy.name)'. This action cannot be undone.")
        }
        .frame(minWidth: 600, minHeight: 700)
    }
}

// MARK: - Helper Views

struct DetailRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label + ":")
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
        .font(.body)
    }
}

#Preview {
    StrategyDetailView(
        strategy: Strategy(
            name: "Test Strategy",
            description: "A test strategy for preview",
            isActive: true,
            mode: "PAPER",
            validated: true
        ),
        viewModel: StrategyViewModel()
    )
}
