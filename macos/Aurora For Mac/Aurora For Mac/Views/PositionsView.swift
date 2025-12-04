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
                List {
                    ForEach(tradeService.openPositions) { position in
                        PositionCard(position: position) {
                            selectedPosition = position
                            showCloseConfirmation = true
                        }
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            Button(role: .destructive) {
                                selectedPosition = position
                                showCloseConfirmation = true
                            } label: {
                                Label("Close", systemImage: "xmark.circle")
                            }
                        }
                    }
                }
                .listStyle(.inset)
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

// MARK: - Preview

struct PositionsView_Previews: PreviewProvider {
    static var previews: some View {
        PositionsView()
            .frame(width: 1000, height: 600)
    }
}
