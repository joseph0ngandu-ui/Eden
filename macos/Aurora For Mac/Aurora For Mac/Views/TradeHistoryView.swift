import SwiftUI

struct TradeHistoryView: View {
    @StateObject private var tradeService = TradeService.shared
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var filterType: FilterType = .all
    @State private var searchText = ""

    enum FilterType: String, CaseIterable {
        case all = "All"
        case profitable = "Profitable"
        case losses = "Losses"
    }

    var filteredTrades: [Trade] {
        var trades = tradeService.tradeHistory

        // Apply filter
        switch filterType {
        case .all:
            break
        case .profitable:
            trades = trades.filter { $0.isProfitable }
        case .losses:
            trades = trades.filter { !$0.isProfitable && $0.realizedPnl != nil }
        }

        // Apply search
        if !searchText.isEmpty {
            trades = trades.filter {
                $0.symbol.lowercased().contains(searchText.lowercased())
                    || $0.strategy?.lowercased().contains(searchText.lowercased()) ?? false
            }
        }

        return trades
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Trade History")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Spacer()

                // Filter Picker
                Picker("Filter", selection: $filterType) {
                    ForEach(FilterType.allCases, id: \.self) { type in
                        Text(type.rawValue).tag(type)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)

                // Refresh Button
                Button {
                    Task {
                        await refreshHistory()
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.title3)
                }
                .buttonStyle(.plain)
            }
            .padding()

            // Search Bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)

                TextField("Search symbol or strategy...", text: $searchText)
                    .textFieldStyle(.plain)

                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)
            .padding(.horizontal)

            Divider()
                .padding(.top)

            // Trade List
            if tradeService.isLoading && tradeService.tradeHistory.isEmpty {
                VStack {
                    Spacer()
                    ProgressView("Loading trade history...")
                    Spacer()
                }
            } else if filteredTrades.isEmpty {
                VStack {
                    Spacer()
                    Image(systemName: "tray")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                    Text(searchText.isEmpty ? "No trade history" : "No matching trades")
                        .font(.title2)
                        .foregroundColor(.secondary)
                        .padding(.top)
                    Spacer()
                }
            } else {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(filteredTrades) { trade in
                            TradeRow(trade: trade)
                        }
                    }
                    .padding()
                }
            }
        }
        .task {
            await loadHistory()
        }
        .alert("Error", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage)
        }
    }

    private func loadHistory() async {
        do {
            try await tradeService.getTradeHistory(limit: 500)
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    private func refreshHistory() async {
        do {
            try await tradeService.getTradeHistory(limit: 500)
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }
}

// MARK: - Trade Row

struct TradeRow: View {
    let trade: Trade

    var body: some View {
        HStack(spacing: 16) {
            // Timestamp
            VStack(alignment: .leading, spacing: 4) {
                Text(trade.timestamp, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(trade.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .frame(width: 100, alignment: .leading)

            // Symbol
            VStack(alignment: .leading, spacing: 4) {
                Text(trade.symbol)
                    .font(.headline)
                    .fontWeight(.semibold)

                if let strategy = trade.strategy {
                    Text(strategy)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .frame(width: 120, alignment: .leading)

            // Side & Type
            VStack(spacing: 4) {
                Text(trade.side.rawValue)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(sideColor.opacity(0.2))
                    .foregroundColor(sideColor)
                    .cornerRadius(4)

                Text(trade.type.rawValue)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            // Quantity
            VStack(alignment: .trailing, spacing: 4) {
                Text("Quantity")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(String(format: "%.4f", trade.quantity))
                    .font(.subheadline)
            }

            // Price
            VStack(alignment: .trailing, spacing: 4) {
                Text("Price")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(String(format: "$%.2f", trade.price))
                    .font(.subheadline)
            }

            // Commission
            VStack(alignment: .trailing, spacing: 4) {
                Text("Fee")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(String(format: "$%.2f", trade.commission))
                    .font(.subheadline)
                    .foregroundColor(.orange)
            }

            Spacer()

            // P&L
            if let pnl = trade.realizedPnl {
                VStack(alignment: .trailing, spacing: 4) {
                    Text("P&L")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text(trade.formattedPnl ?? "")
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundColor(pnlColor)
                }
                .frame(width: 100, alignment: .trailing)
            }
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
        .shadow(radius: 1)
    }

    private var sideColor: Color {
        trade.side == .buy ? .green : .red
    }

    private var pnlColor: Color {
        trade.isProfitable ? .green : .red
    }
}

// MARK: - Preview

struct TradeHistoryView_Previews: PreviewProvider {
    static var previews: some View {
        TradeHistoryView()
            .frame(width: 1200, height: 700)
    }
}
