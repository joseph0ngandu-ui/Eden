import Charts
import SwiftUI

struct PerformanceView: View {
    @StateObject private var performanceService = PerformanceService.shared
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var selectedTab: Tab = .overview

    enum Tab: String, CaseIterable {
        case overview = "Overview"
        case equityCurve = "Equity Curve"
        case daily = "Daily Summary"
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Performance")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Spacer()

                // Tab Picker
                Picker("View", selection: $selectedTab) {
                    ForEach(Tab.allCases, id: \.self) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 400)

                // Refresh Button
                Button {
                    Task {
                        await refreshData()
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.title3)
                }
                .buttonStyle(.plain)
            }
            .padding()

            Divider()

            // Content
            if performanceService.isLoading && performanceService.stats == nil {
                VStack {
                    Spacer()
                    ProgressView("Loading performance data...")
                    Spacer()
                }
            } else {
                ScrollView {
                    switch selectedTab {
                    case .overview:
                        OverviewTab(stats: performanceService.stats)
                    case .equityCurve:
                        EquityCurveTab(points: performanceService.equityCurve)
                    case .daily:
                        DailySummaryTab(summaries: performanceService.dailySummaries)
                    }
                }
            }
        }
        .task {
            await loadData()
        }
        .alert("Error", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage)
        }
    }

    private func loadData() async {
        do {
            try await performanceService.refreshAllData()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    private func refreshData() async {
        do {
            try await performanceService.refreshAllData()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }
}

// MARK: - Overview Tab

struct OverviewTab: View {
    let stats: PerformanceStats?

    var body: some View {
        VStack(spacing: 20) {
            if let stats = stats {
                // Main Metrics
                LazyVGrid(
                    columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible()),
                        GridItem(.flexible()),
                        GridItem(.flexible()),
                    ], spacing: 16
                ) {
                    StatCard(
                        title: "Total Profit",
                        value: stats.formattedTotalProfit,
                        icon: "dollarsign.circle.fill",
                        color: stats.totalProfit >= 0 ? .green : .red
                    )

                    StatCard(
                        title: "Win Rate",
                        value: stats.formattedWinRate,
                        icon: "chart.line.uptrend.xyaxis",
                        color: .blue
                    )

                    StatCard(
                        title: "Profit Factor",
                        value: stats.formattedProfitFactor,
                        icon: "chart.bar.fill",
                        color: .purple
                    )

                    StatCard(
                        title: "Total Trades",
                        value: "\(stats.totalTrades)",
                        icon: "list.bullet",
                        color: .orange
                    )
                }
                .padding()

                // Secondary Metrics
                LazyVGrid(
                    columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible()),
                        GridItem(.flexible()),
                    ], spacing: 16
                ) {
                    DetailCard(
                        title: "Winning Trades",
                        value: "\(stats.winningTrades)",
                        color: .green
                    )

                    DetailCard(
                        title: "Losing Trades",
                        value: "\(stats.losingTrades)",
                        color: .red
                    )

                    DetailCard(
                        title: "Average Win",
                        value: String(format: "$%.2f", stats.averageWin),
                        color: .green
                    )

                    DetailCard(
                        title: "Average Loss",
                        value: String(format: "$%.2f", stats.averageLoss),
                        color: .red
                    )

                    if let sharpe = stats.sharpeRatio {
                        DetailCard(
                            title: "Sharpe Ratio",
                            value: String(format: "%.2f", sharpe),
                            color: .blue
                        )
                    }

                    DetailCard(
                        title: "Max Drawdown",
                        value: stats.formattedMaxDrawdown,
                        color: .orange
                    )
                }
                .padding()

                // ROI if available
                if let roi = stats.roi {
                    StatCard(
                        title: "Return on Investment",
                        value: stats.formattedROI,
                        icon: "percent",
                        color: roi >= 0 ? .green : .red
                    )
                    .padding()
                }
            } else {
                Text("No performance data available")
                    .foregroundColor(.secondary)
                    .padding()
            }
        }
    }
}

// MARK: - Equity Curve Tab

struct EquityCurveTab: View {
    let points: [EquityPoint]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !points.isEmpty {
                Text("Account Equity Over Time")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .padding(.horizontal)
                    .padding(.top)

                Chart(points) { point in
                    LineMark(
                        x: .value("Date", point.timestamp),
                        y: .value("Equity", point.equity)
                    )
                    .foregroundStyle(.blue)
                }
                .frame(height: 300)
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(12)
                .padding(.horizontal)

                Text("Balance vs Equity")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .padding(.horizontal)

                Chart(points) { point in
                    LineMark(
                        x: .value("Date", point.timestamp),
                        y: .value("Balance", point.balance)
                    )
                    .foregroundStyle(.green)

                    LineMark(
                        x: .value("Date", point.timestamp),
                        y: .value("Equity", point.equity)
                    )
                    .foregroundStyle(.blue)
                }
                .frame(height: 300)
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(12)
                .padding(.horizontal)

            } else {
                VStack {
                    Spacer()
                    Text("No equity curve data available")
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .padding(.bottom)
    }
}

// MARK: - Daily Summary Tab

struct DailySummaryTab: View {
    let summaries: [DailySummary]

    var body: some View {
        VStack(spacing: 16) {
            if !summaries.isEmpty {
                ForEach(summaries) { summary in
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(summary.date, style: .date)
                                .font(.headline)
                            Text("\(summary.trades) trades")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        VStack(alignment: .trailing, spacing: 4) {
                            Text(summary.formattedProfit)
                                .font(.headline)
                                .foregroundColor(summary.profit >= 0 ? .green : .red)
                            Text("\(summary.winningTrades)W / \(summary.losingTrades)L")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(8)
                }
                .padding(.horizontal)
            } else {
                VStack {
                    Spacer()
                    Text("No daily summary data available")
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .padding(.vertical)
    }
}

// MARK: - Detail Card

struct DetailCard: View {
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Text(value)
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(color)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Preview

struct PerformanceView_Previews: PreviewProvider {
    static var previews: some View {
        PerformanceView()
            .frame(width: 1200, height: 800)
    }
}
