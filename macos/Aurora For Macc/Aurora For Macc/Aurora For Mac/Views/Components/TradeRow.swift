import SwiftUI

struct TradeRow: View {
    let trade: Trade

    var body: some View {
        HStack(spacing: 12) {
            // Trade type icon
            Image(systemName: tradeIcon)
                .font(.title3)
                .foregroundStyle(tradeColor)
                .frame(width: 30)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(trade.symbol)
                        .font(.headline)
                        .fontWeight(.semibold)

                    Text(trade.side.rawValue)
                        .font(.caption)
                        .fontWeight(.medium)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            trade.side == .buy ? Color.green.opacity(0.2) : Color.red.opacity(0.2)
                        )
                        .foregroundColor(trade.side == .buy ? .green : .red)
                        .cornerRadius(4)

                    Text(trade.type.rawValue.replacingOccurrences(of: "_", with: " "))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                HStack(spacing: 12) {
                    Label("\(String(format: "%.4f", trade.quantity))", systemImage: "number")
                    Label(
                        "$\(String(format: "%.2f", trade.price))", systemImage: "dollarsign.circle")
                }
                .font(.caption)
                .foregroundColor(.secondary)

                if let strategy = trade.strategy {
                    Text(strategy)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .italic()
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text(trade.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(trade.timestamp, style: .date)
                    .font(.caption2)
                    .foregroundColor(.secondary)

                if let pnl = trade.formattedPnl {
                    Text(pnl)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(trade.isProfitable ? .green : .red)
                }
            }
        }
        .padding(.vertical, 8)
    }

    private var tradeIcon: String {
        switch trade.type {
        case .entry:
            return trade.side == .buy ? "arrow.up.circle.fill" : "arrow.down.circle.fill"
        case .exit:
            return "arrow.uturn.left.circle.fill"
        case .stopLoss:
            return "shield.lefthalf.filled"
        case .takeProfit:
            return "star.circle.fill"
        }
    }

    private var tradeColor: Color {
        switch trade.type {
        case .entry:
            return trade.side == .buy ? .green : .red
        case .exit:
            return .blue
        case .stopLoss:
            return .red
        case .takeProfit:
            return .green
        }
    }
}

#Preview {
    List {
        ForEach(Trade.samples) { trade in
            TradeRow(trade: trade)
        }
    }
    .listStyle(.inset)
    .frame(width: 500, height: 400)
}
