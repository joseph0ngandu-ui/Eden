import Foundation

struct Trade: Codable, Identifiable, Hashable {
    let id: String
    let symbol: String
    let side: TradeSide
    let type: TradeType
    let quantity: Double
    let price: Double
    let commission: Double
    let realizedPnl: Double?
    let strategy: String?
    let timestamp: Date
    let orderId: String?

    enum TradeSide: String, Codable {
        case buy = "BUY"
        case sell = "SELL"
    }

    enum TradeType: String, Codable {
        case entry = "ENTRY"
        case exit = "EXIT"
        case stopLoss = "STOP_LOSS"
        case takeProfit = "TAKE_PROFIT"
    }

    var totalValue: Double {
        quantity * price
    }

    var formattedPnl: String? {
        guard let pnl = realizedPnl else { return nil }
        let sign = pnl >= 0 ? "+" : ""
        return "\(sign)$\(String(format: "%.2f", pnl))"
    }

    var isProfitable: Bool {
        realizedPnl ?? 0 > 0
    }

    enum CodingKeys: String, CodingKey {
        case id
        case symbol
        case side
        case type
        case quantity
        case price
        case commission
        case realizedPnl = "realized_pnl"
        case strategy
        case timestamp
        case orderId = "order_id"
    }
}

// Sample data for previews
extension Trade {
    static let sample = Trade(
        id: "trade_001",
        symbol: "BTCUSDT",
        side: .buy,
        type: .entry,
        quantity: 0.05,
        price: 43250.0,
        commission: 2.16,
        realizedPnl: nil,
        strategy: "Momentum Strategy",
        timestamp: Date().addingTimeInterval(-3600),
        orderId: "order_123"
    )

    static let samples: [Trade] = [
        Trade(
            id: "trade_001",
            symbol: "BTCUSDT",
            side: .sell,
            type: .takeProfit,
            quantity: 0.05,
            price: 44100.0,
            commission: 2.21,
            realizedPnl: 42.50,
            strategy: "Momentum Strategy",
            timestamp: Date().addingTimeInterval(-300),
            orderId: "order_125"
        ),
        Trade(
            id: "trade_002",
            symbol: "ETHUSDT",
            side: .buy,
            type: .entry,
            quantity: 1.2,
            price: 2280.0,
            commission: 2.74,
            realizedPnl: nil,
            strategy: "RSI Divergence",
            timestamp: Date().addingTimeInterval(-7200),
            orderId: "order_124"
        ),
        Trade(
            id: "trade_003",
            symbol: "SOLUSDT",
            side: .sell,
            type: .stopLoss,
            quantity: 10.0,
            price: 98.50,
            commission: 0.99,
            realizedPnl: -15.30,
            strategy: "Breakout Strategy",
            timestamp: Date().addingTimeInterval(-10800),
            orderId: "order_122"
        ),
    ]
}
