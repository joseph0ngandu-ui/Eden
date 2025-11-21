import SwiftUI

struct PositionCard: View {
    let position: Position
    var onClose: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                HStack(spacing: 8) {
                    Image(
                        systemName: position.side == .long
                            ? "arrow.up.circle.fill" : "arrow.down.circle.fill"
                    )
                    .foregroundStyle(position.side == .long ? .green : .red)
                    .font(.title3)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(position.symbol)
                            .font(.headline)
                            .fontWeight(.semibold)

                        Text(position.side == .long ? "Long" : "Short")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()

                // Leverage badge
                Text("\(position.leverage)x")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.blue.opacity(0.2))
                    .cornerRadius(4)
            }

            Divider()

            // Position details
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Size")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.4f", position.size))
                        .font(.subheadline)
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Entry")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("$\(String(format: "%.2f", position.entryPrice))")
                        .font(.subheadline)
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Current")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("$\(String(format: "%.2f", position.currentPrice))")
                        .font(.subheadline)
                        .fontWeight(.medium)
                }
            }

            // P&L
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Unrealized P&L")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        Text(position.formattedPnl)
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundColor(position.isProfitable ? .green : .red)

                        Text(position.formattedPnlPercent)
                            .font(.subheadline)
                            .foregroundColor(position.isProfitable ? .green : .red)
                    }
                }

                Spacer()

                if let onClose = onClose {
                    Button(action: onClose) {
                        Label("Close", systemImage: "xmark.circle.fill")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
            }

            // Strategy tag
            if let strategy = position.strategy {
                HStack {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.caption2)
                    Text(strategy)
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
                .padding(.top, 4)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(NSColor.controlBackgroundColor))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
}

#Preview {
    VStack(spacing: 16) {
        PositionCard(position: Position.sample)
        PositionCard(position: Position.samples[1])
    }
    .padding()
    .frame(width: 400)
}
