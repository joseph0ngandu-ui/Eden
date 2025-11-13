//
//  SharedDataService.swift
//  Eden
//
//  Shared data service for communication between app and widgets
//  Uses App Groups to share data with widget extensions
//

import Foundation
import WidgetKit

class SharedDataService {
    static let shared = SharedDataService()
    
    // MARK: - App Group Configuration
    // NOTE: You need to add an App Group in Xcode:
    // 1. Select your target -> Signing & Capabilities
    // 2. Add "App Groups" capability
    // 3. Add group: group.com.eden.trading
    private let appGroupIdentifier = "group.com.eden.trading"
    
    private var userDefaults: UserDefaults? {
        return UserDefaults(suiteName: appGroupIdentifier)
    }
    
    private init() {}
    
    // MARK: - Widget Data Model
    struct WidgetData: Codable {
        let isRunning: Bool
        let balance: Double
        let dailyPnL: Double
        let activePositions: Int
        let winRate: Double
        let totalTrades: Int
        let profitFactor: Double
        let riskTier: String
        let lastUpdate: Date
        
        // Computed properties for widget display
        var isInProfit: Bool {
            return dailyPnL > 0
        }
        
        var statusText: String {
            return isRunning ? "ACTIVE" : "STOPPED"
        }
        
        var profitStatusText: String {
            return isInProfit ? "IN PROFIT" : (dailyPnL < 0 ? "IN LOSS" : "BREAK EVEN")
        }
    }
    
    // MARK: - Save Widget Data
    func saveWidgetData(_ data: WidgetData) {
        guard let defaults = userDefaults else {
            print("⚠️ Failed to access App Group UserDefaults")
            return
        }
        
        do {
            let encoder = JSONEncoder()
            let encoded = try encoder.encode(data)
            defaults.set(encoded, forKey: "widgetData")
            defaults.synchronize()
            
            // Reload all widgets to show updated data
            WidgetCenter.shared.reloadAllTimelines()
            
            print("✓ Widget data saved and reloaded")
        } catch {
            print("✗ Failed to encode widget data: \(error)")
        }
    }
    
    // MARK: - Load Widget Data
    func loadWidgetData() -> WidgetData? {
        guard let defaults = userDefaults,
              let data = defaults.data(forKey: "widgetData") else {
            return nil
        }
        
        do {
            let decoder = JSONDecoder()
            let widgetData = try decoder.decode(WidgetData.self, from: data)
            return widgetData
        } catch {
            print("✗ Failed to decode widget data: \(error)")
            return nil
        }
    }
    
    // MARK: - Convenience Save Methods
    func saveBotStatus(
        isRunning: Bool,
        balance: Double,
        dailyPnL: Double,
        activePositions: Int,
        winRate: Double,
        totalTrades: Int,
        profitFactor: Double,
        riskTier: String
    ) {
        let data = WidgetData(
            isRunning: isRunning,
            balance: balance,
            dailyPnL: dailyPnL,
            activePositions: activePositions,
            winRate: winRate,
            totalTrades: totalTrades,
            profitFactor: profitFactor,
            riskTier: riskTier,
            lastUpdate: Date()
        )
        saveWidgetData(data)
    }
    
    // MARK: - Widget Reload
    func reloadWidgets() {
        WidgetCenter.shared.reloadAllTimelines()
    }
    
    func reloadSpecificWidget(kind: String) {
        WidgetCenter.shared.reloadTimelines(ofKind: kind)
    }
}

// MARK: - Widget Data Extensions
extension SharedDataService.WidgetData {
    // Format balance with currency symbol
    var formattedBalance: String {
        return String(format: "$%.2f", balance)
    }
    
    // Format daily P&L with sign and color indicator
    var formattedDailyPnL: String {
        let sign = dailyPnL >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, dailyPnL)
    }
    
    // Format win rate as percentage
    var formattedWinRate: String {
        return String(format: "%.1f%%", winRate)
    }
    
    // Format profit factor
    var formattedProfitFactor: String {
        return String(format: "%.2fx", profitFactor)
    }
    
    // Time since last update
    var timeSinceUpdate: String {
        let interval = Date().timeIntervalSince(lastUpdate)
        if interval < 60 {
            return "Just now"
        } else if interval < 3600 {
            let minutes = Int(interval / 60)
            return "\(minutes)m ago"
        } else {
            let hours = Int(interval / 3600)
            return "\(hours)h ago"
        }
    }
}
