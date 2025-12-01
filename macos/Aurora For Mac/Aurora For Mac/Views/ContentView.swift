import SwiftUI

struct ContentView: View {
    @EnvironmentObject var authService: AuthService
    @State private var selectedTab: Tab = .strategies

    enum Tab {
        case botControl
        case positions
        case trades
        case performance
        case strategies
        case training
        case backtest
        case monitor
    }

    var body: some View {
        NavigationSplitView {
            // Sidebar
            List(selection: $selectedTab) {
                Section("Trading") {
                    Label("Bot Control", systemImage: "play.circle.fill")
                        .tag(Tab.botControl)

                    Label("Positions", systemImage: "chart.line.uptrend.xyaxis")
                        .tag(Tab.positions)

                    Label("Trades", systemImage: "list.bullet")
                        .tag(Tab.trades)

                    Label("Performance", systemImage: "chart.bar.fill")
                        .tag(Tab.performance)
                }

                Section("Strategy") {
                    Label("Strategies", systemImage: "doc.text.fill")
                        .tag(Tab.strategies)

                    Label("ML Training", systemImage: "brain.head.profile")
                        .tag(Tab.training)

                    Label("Backtest", systemImage: "clock.arrow.circlepath")
                        .tag(Tab.backtest)
                }

                Section("Monitoring") {
                    Label("Monitor", systemImage: "chart.xyaxis.line")
                        .tag(Tab.monitor)
                }
            }
            .navigationTitle("Aurora")
            .navigationSplitViewColumnWidth(min: 200, ideal: 200)
        } detail: {
            // Detail view based on selection
            Group {
                switch selectedTab {
                case .botControl:
                    BotControlView()
                case .positions:
                    PositionsView()
                case .trades:
                    TradeHistoryView()
                case .performance:
                    PerformanceView()
                case .strategies:
                    StrategyListView()
                case .training:
                    MLTrainingView()
                case .backtest:
                    BacktestView()
                case .monitor:
                    MonitorView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .sheet(isPresented: .constant(!authService.isAuthenticated)) {
            LoginView()
                .environmentObject(authService)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AuthService.shared)
        .environmentObject(StrategyViewModel())
}
