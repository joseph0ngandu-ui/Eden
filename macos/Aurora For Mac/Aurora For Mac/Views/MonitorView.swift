import SwiftUI

struct MonitorView: View {
    @StateObject private var wsService = WebSocketService.shared
    @State private var autoRefresh: Bool = true
    @State private var selectedFilter: PositionFilter = .all
    @State private var totalPnL: Double = 0.0

    enum PositionFilter: String, CaseIterable {
        case all = "All"
        case long = "Long"
        case short = "Short"
        case profitable = "Profitable"
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                HStack {
                    VStack(alignment: .leading) {
                        Text("Monitor")
                            .font(.title)
                            .fontWeight(.bold)

                        HStack(spacing: 8) {
                            Circle()
                                .fill(wsService.isConnected ? Color.green : Color.red)
                                .frame(width: 8, height: 8)

                            Text(wsService.isConnected ? "Connected" : "Disconnected")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }

                    Spacer()

                    // Auto-refresh toggle
                    Toggle(isOn: $autoRefresh) {
                        Label("Auto-refresh", systemImage: "arrow.clockwise")
                    }
                    .toggleStyle(.switch)

                    // Reconnect button
                    Button {
                        if wsService.isConnected {
                            wsService.disconnect()
                        } else {
                            wsService.connect()
                        }
                    } label: {
                        Label(
                            wsService.isConnected ? "Disconnect" : "Connect",
                            systemImage: wsService.isConnected ? "stop.circle" : "play.circle"
                        )
                    }
                    .buttonStyle(.bordered)
                }
                .padding()

                // Connection error
                if let error = wsService.connectionError {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text(error)
                        Spacer()
                    }
                    .padding()
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
                    .padding(.horizontal)
                }

                // Statistics
                HStack(spacing: 16) {
                    StatCard(
                        title: "Total P&L",
                        value: formatPnL(totalPnL),
                        icon: "dollarsign.circle.fill",
                        color: totalPnL >= 0 ? .green : .red,
                        trend: totalPnL >= 0
                            ? "+\(String(format: "%.1f", abs(totalPnL)))"
                            : "-\(String(format: "%.1f", abs(totalPnL)))"
                    )

                    StatCard(
                        title: "Active Positions",
                        value: "\(filteredPositions.count)",
                        icon: "chart.line.uptrend.xyaxis",
                        color: .blue
                    )

                    StatCard(
                        title: "Recent Trades",
                        value: "\(wsService.recentTrades.count)",
                        icon: "arrow.left.arrow.right.circle.fill",
                        color: .orange
                    )
                }
                .padding(.horizontal)

                // Positions section
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Active Positions")
                            .font(.title2)
                            .fontWeight(.semibold)

                        Spacer()

                        // Filter picker
                        Picker("Filter", selection: $selectedFilter) {
                            ForEach(PositionFilter.allCases, id: \.self) { filter in
                                Text(filter.rawValue).tag(filter)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 300)
                    }
                    .padding(.horizontal)

                    if filteredPositions.isEmpty {
                        ContentUnavailableView {
                            Label("No Active Positions", systemImage: "chart.xyaxis.line")
                        } description: {
                            Text(
                                "Your active positions will appear here when strategies open trades."
                            )
                        }
                        .frame(height: 200)
                    } else {
                        LazyVStack(spacing: 12) {
                            ForEach(filteredPositions) { position in
                                PositionCard(position: position) {
                                    // Handle close position
                                    closePosition(position)
                                }
                            }
                        }
                        .padding(.horizontal)
                    }
                }
                .padding(.vertical)

                // Recent trades section
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Recent Trades")
                            .font(.title2)
                            .fontWeight(.semibold)

                        Spacer()

                        Button {
                            // TODO: Show full trade history
                        } label: {
                            Text("View All")
                                .font(.subheadline)
                        }
                    }
                    .padding(.horizontal)

                    if wsService.recentTrades.isEmpty {
                        ContentUnavailableView {
                            Label("No Recent Trades", systemImage: "clock")
                        } description: {
                            Text("Trade history will appear here as your strategies execute.")
                        }
                        .frame(height: 200)
                    } else {
                        VStack(spacing: 0) {
                            ForEach(wsService.recentTrades.prefix(10)) { trade in
                                TradeRow(trade: trade)
                                    .padding(.horizontal)

                                if trade.id != wsService.recentTrades.prefix(10).last?.id {
                                    Divider()
                                        .padding(.horizontal)
                                }
                            }
                        }
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color(NSColor.controlBackgroundColor))
                        )
                        .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
        }
        .onAppear {
            wsService.connect()
        }
        .onDisappear {
            if !autoRefresh {
                wsService.disconnect()
            }
        }
        .onChange(of: wsService.positions) { _, newPositions in
            calculateTotalPnL(from: newPositions)
        }
    }

    private var filteredPositions: [Position] {
        switch selectedFilter {
        case .all:
            return wsService.positions
        case .long:
            return wsService.positions.filter { $0.side == .long }
        case .short:
            return wsService.positions.filter { $0.side == .short }
        case .profitable:
            return wsService.positions.filter { $0.isProfitable }
        }
    }

    private func calculateTotalPnL(from positions: [Position]) {
        totalPnL = positions.reduce(0) { $0 + $1.unrealizedPnl }
    }

    private func formatPnL(_ value: Double) -> String {
        let sign = value >= 0 ? "+" : ""
        return "\(sign)$\(String(format: "%.2f", value))"
    }

    private func closePosition(_ position: Position) {
        // TODO: Implement close position API call
        print("Closing position: \(position.symbol)")
    }
}

#Preview {
    MonitorView()
}
