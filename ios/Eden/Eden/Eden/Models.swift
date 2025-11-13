//
//  Models.swift
//  Eden
//
//  All data models for the Eden trading bot app
//

import Foundation

// MARK: - Position Model

/// Represents an active trading position
struct Position: Identifiable, Codable {
    var id = UUID()
    let symbol: String
    let direction: String  // "LONG" or "SHORT"
    let entry: Double
    let current: Double
    let pnl: Double
    let confidence: Double  // 0.0 to 1.0
    let bars: Int
    
    enum CodingKeys: String, CodingKey {
        case symbol, direction, entry, current, pnl, confidence, bars
    }
}

// MARK: - Trade Model

/// Represents a completed trade
struct Trade: Identifiable, Codable {
    var id = UUID()
    let symbol: String
    let pnl: Double
    let time: String
    let rValue: Double  // R-multiple
    
    enum CodingKeys: String, CodingKey {
        case symbol, pnl, time, rValue
    }
}

// MARK: - Equity Point Model

/// Represents a data point on the equity curve
struct EquityPoint: Identifiable, Codable {
    var id = UUID()
    let time: String
    let value: Double
    
    enum CodingKeys: String, CodingKey {
        case time, value
    }
}

// MARK: - Bot Status Model

/// Bot status information from API
struct BotStatus: Codable {
    let isRunning: Bool
    let balance: Double
    let dailyPnL: Double
    let activePositions: Int
    let winRate: Double
    let riskTier: String
    let totalTrades: Int?
    let profitFactor: Double?
    let peakBalance: Double?
    let currentDrawdown: Double?
    
    enum CodingKeys: String, CodingKey {
        case isRunning = "is_running"
        case balance
        case dailyPnL = "daily_pnl"
        case activePositions = "active_positions"
        case winRate = "win_rate"
        case riskTier = "risk_tier"
        case totalTrades = "total_trades"
        case profitFactor = "profit_factor"
        case peakBalance = "peak_balance"
        case currentDrawdown = "current_drawdown"
    }
}

// MARK: - Strategy Model

/// Trading strategy information
struct Strategy: Identifiable, Codable {
    let id: Int
    let name: String
    let description: String?
    let isActive: Bool
    let winRate: Double?
    let profitFactor: Double?
    let totalTrades: Int?
    let category: String?
    let riskLevel: String?
    let createdAt: String?
    
    enum CodingKeys: String, CodingKey {
        case id, name, description
        case isActive = "is_active"
        case winRate = "win_rate"
        case profitFactor = "profit_factor"
        case totalTrades = "total_trades"
        case category
        case riskLevel = "risk_level"
        case createdAt = "created_at"
    }
}

// MARK: - Performance Metrics Model

/// Detailed performance analytics
struct PerformanceMetrics: Codable {
    let peakBalance: Double
    let currentDrawdown: Double
    let maxDrawdown: Double
    let totalTrades: Int
    let winningTrades: Int
    let losingTrades: Int
    let winRate: Double
    let profitFactor: Double
    let sharpeRatio: Double?
    let averageWin: Double
    let averageLoss: Double
    let largestWin: Double
    let largestLoss: Double
    let consecutiveWins: Int
    let consecutiveLosses: Int
    
    enum CodingKeys: String, CodingKey {
        case peakBalance = "peak_balance"
        case currentDrawdown = "current_drawdown"
        case maxDrawdown = "max_drawdown"
        case totalTrades = "total_trades"
        case winningTrades = "winning_trades"
        case losingTrades = "losing_trades"
        case winRate = "win_rate"
        case profitFactor = "profit_factor"
        case sharpeRatio = "sharpe_ratio"
        case averageWin = "average_win"
        case averageLoss = "average_loss"
        case largestWin = "largest_win"
        case largestLoss = "largest_loss"
        case consecutiveWins = "consecutive_wins"
        case consecutiveLosses = "consecutive_losses"
    }
}

// MARK: - MT5 Account Model (Simple version for Views)

/// MT5 account display model
struct MT5Account: Identifiable {
    let id = UUID()
    let accountNumber: String
    let accountName: String
    let broker: String
    let server: String
    let isPrimary: Bool
    let isActive: Bool
}

// MARK: - API Response Wrappers

/// Generic success response
struct SuccessResponse: Codable {
    let message: String
    let status: String?
}

/// Generic error response
struct ErrorResponse: Codable {
    let error: String
    let details: String?
    let code: String?
}

// MARK: - WebSocket Messages

/// WebSocket message types
enum WebSocketMessageType: String, Codable {
    case positionUpdate = "position_update"
    case tradeExecuted = "trade_executed"
    case balanceUpdate = "balance_update"
    case botStatus = "bot_status"
    case error = "error"
}

/// WebSocket message wrapper
struct WebSocketMessage: Codable {
    let type: WebSocketMessageType
    let data: String  // JSON string that needs to be decoded based on type
    let timestamp: String?
}
