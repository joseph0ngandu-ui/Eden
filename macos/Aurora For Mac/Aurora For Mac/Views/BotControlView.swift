import SwiftUI

struct BotControlView: View {
    @StateObject private var botService = BotService.shared
    @State private var showError = false
    @State private var errorMessage = ""

    var body: some View {
        VStack(spacing: 20) {
            // Header
            Text("Bot Control")
                .font(.largeTitle)
                .fontWeight(.bold)

            // Status Card
            if let status = botService.botStatus {
                StatusCard(status: status)
                    .transition(.scale)
            } else {
                ProgressView("Loading bot status...")
                    .frame(height: 200)
            }

            // Control Buttons
            HStack(spacing: 16) {
                BotControlButton(
                    title: "Start",
                    icon: "play.fill",
                    color: .green,
                    isEnabled: botService.botStatus?.isStopped ?? false
                ) {
                    await startBot()
                }

                BotControlButton(
                    title: "Pause",
                    icon: "pause.fill",
                    color: .orange,
                    isEnabled: botService.botStatus?.isRunning ?? false
                ) {
                    await pauseBot()
                }

                BotControlButton(
                    title: "Stop",
                    icon: "stop.fill",
                    color: .red,
                    isEnabled: botService.botStatus?.isRunning ?? false
                        || botService.botStatus?.isPaused ?? false
                ) {
                    await stopBot()
                }
            }
            .padding(.horizontal)

            Spacer()
        }
        .padding()
        .task {
            await loadBotStatus()
            botService.startAutoRefresh(interval: 5.0)
        }
        .onDisappear {
            botService.stopAutoRefresh()
        }
        .alert("Error", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage)
        }
    }

    private func loadBotStatus() async {
        do {
            try await botService.getBotStatus()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    private func startBot() async {
        do {
            try await botService.startBot()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    private func pauseBot() async {
        do {
            try await botService.pauseBot()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    private func stopBot() async {
        do {
            try await botService.stopBot()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }
}

// MARK: - Status Card

struct StatusCard: View {
    let status: BotStatus

    var body: some View {
        VStack(spacing: 16) {
            // Status Indicator
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 16, height: 16)

                Text(status.status)
                    .font(.title2)
                    .fontWeight(.semibold)

                Spacer()

                Text(status.uptimeFormatted)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Divider()

            // Metrics
            HStack(spacing: 32) {
                MetricView(
                    title: "Active Trades",
                    value: "\(status.activeTrades)",
                    icon: "chart.line.uptrend.xyaxis"
                )

                MetricView(
                    title: "Today's Profit",
                    value: String(format: "$%.2f", status.todayProfit),
                    icon: "dollarsign.circle",
                    color: status.todayProfit >= 0 ? .green : .red
                )

                MetricView(
                    title: "Total Profit",
                    value: String(format: "$%.2f", status.totalProfit),
                    icon: "chart.bar.fill",
                    color: status.totalProfit >= 0 ? .green : .red
                )
            }
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    private var statusColor: Color {
        switch status.status {
        case "RUNNING": return .green
        case "PAUSED": return .orange
        case "STOPPED": return .red
        default: return .gray
        }
    }
}

// MARK: - Metric View

struct MetricView: View {
    let title: String
    let value: String
    let icon: String
    var color: Color = .blue

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)

            Text(value)
                .font(.title3)
                .fontWeight(.bold)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Control Button

struct BotControlButton: View {
    let title: String
    let icon: String
    let color: Color
    let isEnabled: Bool
    let action: () async -> Void

    @State private var isProcessing = false

    var body: some View {
        Button {
            Task {
                isProcessing = true
                await action()
                isProcessing = false
            }
        } label: {
            VStack(spacing: 8) {
                if isProcessing {
                    ProgressView()
                        .scaleEffect(0.8)
                } else {
                    Image(systemName: icon)
                        .font(.title)
                }

                Text(title)
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 80)
            .background(isEnabled ? color.opacity(0.2) : Color.gray.opacity(0.1))
            .foregroundColor(isEnabled ? color : .gray)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isEnabled ? color : Color.gray.opacity(0.3), lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled || isProcessing)
    }
}

// MARK: - Preview

struct BotControlView_Previews: PreviewProvider {
    static var previews: some View {
        BotControlView()
            .frame(width: 800, height: 600)
    }
}
