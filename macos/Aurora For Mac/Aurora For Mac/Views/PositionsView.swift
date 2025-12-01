import SwiftUI

struct PositionsView: View {
    @StateObject private var tradeService = TradeService.shared
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var selectedPosition: Position?
    @State private var showCloseConfirmation = false

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Open Positions")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Spacer()

                Button {
                    Task {
                        await refreshPositions()
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.title3)
                }
                .buttonStyle(.plain)
            }
            .padding()

            Divider()

            // Positions List
            if tradeService.isLoading && tradeService.openPositions.isEmpty {
                VStack {
                    Spacer()
                    ProgressView("Loading positions...")
                    Spacer()
                }
            } else if tradeService.openPositions.isEmpty {
                VStack {
                    Spacer()
                    Image(systemName: "tray")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                    Text("No open positions")
                        .font(.title2)
                        .foregroundColor(.secondary)
                        .padding(.top)
                    Spacer()
                }
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(tradeService.openPositions) { position in
                            PositionCard(position: position) {
                                selectedPosition = position
                                showCloseConfirmation = true
                            }
                        }
                    }
                    .padding()
                }
            }
        }
        .task {
            await loadPositions()
            tradeService.startAutoRefresh(interval: 3.0)
        }
        .onDisappear {
            tradeService.stopAutoRefresh()
        }
        .alert("Error", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage)
        }
        .alert("Close Position", isPresented: $showCloseConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Close Position", role: .destructive) {
                if let position = selectedPosition {
                    Task {
                        await closePosition(position)
                    }
                }
            }
        } message: {
            if let position = selectedPosition {
                Text(
                    "Are you sure you want to close \(position.symbol) \(position.side.rawValue) position?"
                )
            }
        }
    }

    private func loadPositions() async {
        do {
            try await tradeService.getOpenPositions()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    private func refreshPositions() async {
        do {
            try await tradeService.getOpenPositions()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    private func closePosition(_ position: Position) async {
        do {
            try await tradeService.closeTrade(positionId: position.id)
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }
}

// MARK: - Position Card

struct PositionCard: View {
    let position: Position
    let onClose: () -> Void

    var body: some View {
        HStack(spacing: 16) {
            // Symbol and Side
            VStack(alignment: .leading, spacing: 4) {
                Text(position.symbol)
                    .font(.title3)
                    .fontWeight(.bold)

                HStack(spacing: 4) {
                    Text(position.side.rawValue)
                        .font(.caption)
                        .fontWeight(.semibold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(sideColor.opacity(0.2))
                        .foregroundColor(sideColor)
                        .cornerRadius(4)

                    if let strategy = position.strategy {
                        Text(strategy)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            Spacer()

            // Entry Price
            VStack(alignment: .trailing, spacing: 4) {
                Text("Entry")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(String(format: "$%.2f", position.entryPrice))
                    .font(.subheadline)
                    .fontWeight(.medium)
            }

            // Current Price
            VStack(alignment: .trailing, spacing: 4) {
                Text("Current")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(String(format: "$%.2f", position.currentPrice))
                    .font(.subheadline)
                    .fontWeight(.medium)
            }

            // Size
            VStack(alignment: .trailing, spacing: 4) {
                Text("Size")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(String(format: "%.4f", position.size))
                    .font(.subheadline)
                    .fontWeight(.medium)
            }

            // P&L
            VStack(alignment: .trailing, spacing: 4) {
                Text("P&L")
                    .font(.caption)
                    .foregroundColor(.secondary)

                VStack(alignment: .trailing, spacing: 2) {
                    Text(position.formattedPnl)
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundColor(pnlColor)

                    Text(position.formattedPnlPercent)
                        .font(.caption)
                        .foregroundColor(pnlColor)
                }
            }

            // Close Button
            Button {
                onClose()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(.red)
            }
            .buttonStyle(.plain)
            .help("Close position")
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(10)
        .shadow(radius: 1)
    }

    private var sideColor: Color {
        position.side == .long ? .green : .red
    }

    private var pnlColor: Color {
        position.isProfitable ? .green : .red
    }
}

// MARK: - Preview

struct PositionsView_Previews: PreviewProvider {
    static var previews: some View {
        PositionsView()
            .frame(width: 1000, height: 600)
    }
}
