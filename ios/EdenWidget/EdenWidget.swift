//
//  EdenWidget.swift
//  EdenWidget
//
//  Eden Trading Bot Widgets - Lock Screen & Home Screen
//  Displays bot status, profit/loss, and trading metrics
//

import WidgetKit
import SwiftUI

// MARK: - Widget Entry
struct EdenWidgetEntry: TimelineEntry {
    let date: Date
    let widgetData: SharedDataService.WidgetData?
    
    var isPlaceholder: Bool {
        return widgetData == nil
    }
}

// MARK: - Timeline Provider
struct EdenWidgetProvider: TimelineProvider {
    
    func placeholder(in context: Context) -> EdenWidgetEntry {
        EdenWidgetEntry(
            date: Date(),
            widgetData: SharedDataService.WidgetData(
                isRunning: true,
                balance: 347.82,
                dailyPnL: 23.45,
                activePositions: 2,
                winRate: 68.5,
                totalTrades: 127,
                profitFactor: 2.34,
                riskTier: "AGGRESSIVE",
                lastUpdate: Date()
            )
        )
    }
    
    func getSnapshot(in context: Context, completion: @escaping (EdenWidgetEntry) -> Void) {
        let data = SharedDataService.shared.loadWidgetData()
        let entry = EdenWidgetEntry(date: Date(), widgetData: data)
        completion(entry)
    }
    
    func getTimeline(in context: Context, completion: @escaping (Timeline<EdenWidgetEntry>) -> Void) {
        let currentDate = Date()
        let data = SharedDataService.shared.loadWidgetData()
        
        // Create entry with current data
        let entry = EdenWidgetEntry(date: currentDate, widgetData: data)
        
        // Update widget every 5 minutes
        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 5, to: currentDate)!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        
        completion(timeline)
    }
}

// MARK: - Widget Configuration
@main
struct EdenWidgetBundle: WidgetBundle {
    var body: some Widget {
        EdenStatusWidget()
        EdenLockScreenWidget()
    }
}

// MARK: - Home Screen Widget
struct EdenStatusWidget: Widget {
    let kind: String = "EdenStatusWidget"
    
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: EdenWidgetProvider()) { entry in
            EdenWidgetView(entry: entry)
        }
        .configurationDisplayName("Eden Bot Status")
        .description("View your trading bot status, balance, and performance metrics")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
        .contentMarginsDisabled() // iOS 17+
    }
}

// MARK: - Lock Screen Widget
struct EdenLockScreenWidget: Widget {
    let kind: String = "EdenLockScreenWidget"
    
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: EdenWidgetProvider()) { entry in
            EdenLockScreenWidgetView(entry: entry)
        }
        .configurationDisplayName("Eden Bot Lock Screen")
        .description("Quick view of bot status and profit on your lock screen")
        .supportedFamilies([
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryInline
        ])
    }
}

// MARK: - Home Screen Widget Views
struct EdenWidgetView: View {
    var entry: EdenWidgetEntry
    @Environment(\.widgetFamily) var widgetFamily
    
    var body: some View {
        Group {
            switch widgetFamily {
            case .systemSmall:
                SmallWidgetView(entry: entry)
            case .systemMedium:
                MediumWidgetView(entry: entry)
            case .systemLarge:
                LargeWidgetView(entry: entry)
            default:
                SmallWidgetView(entry: entry)
            }
        }
    }
}

// MARK: - Small Widget (Home Screen)
struct SmallWidgetView: View {
    var entry: EdenWidgetEntry
    
    var body: some View {
        if let data = entry.widgetData {
            ZStack {
                LinearGradient(
                    colors: [Color(red: 0.1, green: 0.1, blue: 0.15), Color(red: 0.05, green: 0.05, blue: 0.1)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                
                VStack(alignment: .leading, spacing: 8) {
                    // Status indicator
                    HStack {
                        Circle()
                            .fill(data.isRunning ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        Text(data.statusText)
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(data.isRunning ? .green : .red)
                        Spacer()
                    }
                    
                    Spacer()
                    
                    // Balance
                    VStack(alignment: .leading, spacing: 2) {
                        Text("BALANCE")
                            .font(.system(size: 9, weight: .medium))
                            .foregroundColor(.gray)
                        Text(data.formattedBalance)
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)
                    }
                    
                    // Daily P&L
                    HStack(spacing: 4) {
                        Image(systemName: data.isInProfit ? "arrow.up.right" : "arrow.down.right")
                            .font(.system(size: 10))
                        Text(data.formattedDailyPnL)
                            .font(.system(size: 14, weight: .semibold))
                    }
                    .foregroundColor(data.isInProfit ? .green : .red)
                    
                    // Last update
                    Text(data.timeSinceUpdate)
                        .font(.system(size: 8))
                        .foregroundColor(.gray)
                }
                .padding(12)
            }
        } else {
            PlaceholderView(text: "No Data")
        }
    }
}

// MARK: - Medium Widget (Home Screen)
struct MediumWidgetView: View {
    var entry: EdenWidgetEntry
    
    var body: some View {
        if let data = entry.widgetData {
            ZStack {
                LinearGradient(
                    colors: [Color(red: 0.1, green: 0.1, blue: 0.15), Color(red: 0.05, green: 0.05, blue: 0.1)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                
                HStack(spacing: 16) {
                    // Left side - Main metrics
                    VStack(alignment: .leading, spacing: 10) {
                        // Status
                        HStack(spacing: 6) {
                            Circle()
                                .fill(data.isRunning ? Color.green : Color.red)
                                .frame(width: 10, height: 10)
                            Text(data.statusText)
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(data.isRunning ? .green : .red)
                        }
                        
                        Spacer()
                        
                        // Balance
                        VStack(alignment: .leading, spacing: 2) {
                            Text("BALANCE")
                                .font(.system(size: 9, weight: .medium))
                                .foregroundColor(.gray)
                            Text(data.formattedBalance)
                                .font(.system(size: 28, weight: .bold))
                                .foregroundColor(.white)
                        }
                        
                        // Daily P&L
                        HStack(spacing: 4) {
                            Image(systemName: data.isInProfit ? "arrow.up.right" : "arrow.down.right")
                                .font(.system(size: 12))
                            Text(data.formattedDailyPnL)
                                .font(.system(size: 16, weight: .semibold))
                        }
                        .foregroundColor(data.isInProfit ? .green : .red)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    
                    // Divider
                    Rectangle()
                        .fill(Color.white.opacity(0.1))
                        .frame(width: 1)
                    
                    // Right side - Stats
                    VStack(alignment: .leading, spacing: 12) {
                        StatRow(label: "Win Rate", value: data.formattedWinRate)
                        StatRow(label: "Positions", value: "\(data.activePositions)")
                        StatRow(label: "Profit Factor", value: data.formattedProfitFactor)
                        StatRow(label: "Risk", value: data.riskTier)
                        
                        Spacer()
                        
                        Text(data.timeSinceUpdate)
                            .font(.system(size: 8))
                            .foregroundColor(.gray)
                    }
                }
                .padding(16)
            }
        } else {
            PlaceholderView(text: "No Data")
        }
    }
}

// MARK: - Large Widget (Home Screen)
struct LargeWidgetView: View {
    var entry: EdenWidgetEntry
    
    var body: some View {
        if let data = entry.widgetData {
            ZStack {
                LinearGradient(
                    colors: [Color(red: 0.1, green: 0.1, blue: 0.15), Color(red: 0.05, green: 0.05, blue: 0.1)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("EDEN TRADING BOT")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.gray)
                            
                            HStack(spacing: 6) {
                                Circle()
                                    .fill(data.isRunning ? Color.green : Color.red)
                                    .frame(width: 10, height: 10)
                                Text(data.statusText)
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(data.isRunning ? .green : .red)
                            }
                        }
                        
                        Spacer()
                        
                        Text(data.timeSinceUpdate)
                            .font(.system(size: 9))
                            .foregroundColor(.gray)
                    }
                    
                    // Balance Section
                    VStack(alignment: .leading, spacing: 4) {
                        Text("BALANCE")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.gray)
                        Text(data.formattedBalance)
                            .font(.system(size: 36, weight: .bold))
                            .foregroundColor(.white)
                        
                        HStack(spacing: 6) {
                            Image(systemName: data.isInProfit ? "arrow.up.right" : "arrow.down.right")
                                .font(.system(size: 14))
                            Text(data.formattedDailyPnL)
                                .font(.system(size: 18, weight: .semibold))
                            Text("today")
                                .font(.system(size: 12))
                                .foregroundColor(.gray)
                        }
                        .foregroundColor(data.isInProfit ? .green : .red)
                    }
                    
                    // Divider
                    Rectangle()
                        .fill(Color.white.opacity(0.1))
                        .frame(height: 1)
                    
                    // Performance Metrics
                    HStack(spacing: 20) {
                        VStack(alignment: .leading, spacing: 8) {
                            MetricBox(label: "Win Rate", value: data.formattedWinRate, color: .green)
                            MetricBox(label: "Profit Factor", value: data.formattedProfitFactor, color: .blue)
                        }
                        
                        VStack(alignment: .leading, spacing: 8) {
                            MetricBox(label: "Active Positions", value: "\(data.activePositions)", color: .orange)
                            MetricBox(label: "Total Trades", value: "\(data.totalTrades)", color: .purple)
                        }
                    }
                    
                    // Risk Tier
                    HStack {
                        Text("RISK TIER:")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.gray)
                        Text(data.riskTier)
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(.yellow)
                        
                        Spacer()
                        
                        Text(data.profitStatusText)
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(data.isInProfit ? .green : .red)
                    }
                }
                .padding(16)
            }
        } else {
            PlaceholderView(text: "No Data Available")
        }
    }
}

// MARK: - Lock Screen Widget Views
struct EdenLockScreenWidgetView: View {
    var entry: EdenWidgetEntry
    @Environment(\.widgetFamily) var widgetFamily
    
    var body: some View {
        Group {
            switch widgetFamily {
            case .accessoryCircular:
                CircularLockScreenView(entry: entry)
            case .accessoryRectangular:
                RectangularLockScreenView(entry: entry)
            case .accessoryInline:
                InlineLockScreenView(entry: entry)
            default:
                CircularLockScreenView(entry: entry)
            }
        }
    }
}

// MARK: - Circular Lock Screen Widget
struct CircularLockScreenView: View {
    var entry: EdenWidgetEntry
    
    var body: some View {
        if let data = entry.widgetData {
            ZStack {
                AccessoryWidgetBackground()
                
                VStack(spacing: 2) {
                    // Status indicator
                    Image(systemName: data.isRunning ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .font(.system(size: 18))
                        .foregroundColor(data.isRunning ? .green : .red)
                    
                    // Profit indicator
                    Text(data.isInProfit ? "↑" : "↓")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundColor(data.isInProfit ? .green : .red)
                }
            }
        } else {
            ZStack {
                AccessoryWidgetBackground()
                Text("?")
                    .font(.system(size: 24, weight: .bold))
            }
        }
    }
}

// MARK: - Rectangular Lock Screen Widget
struct RectangularLockScreenView: View {
    var entry: EdenWidgetEntry
    
    var body: some View {
        if let data = entry.widgetData {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Image(systemName: data.isRunning ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .font(.system(size: 12))
                        .foregroundColor(data.isRunning ? .green : .red)
                    Text(data.statusText)
                        .font(.system(size: 12, weight: .bold))
                }
                
                HStack {
                    Text(data.formattedBalance)
                        .font(.system(size: 16, weight: .bold))
                    Spacer()
                    Text(data.formattedDailyPnL)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(data.isInProfit ? .green : .red)
                }
            }
        } else {
            Text("No Data")
                .font(.system(size: 12))
        }
    }
}

// MARK: - Inline Lock Screen Widget
struct InlineLockScreenView: View {
    var entry: EdenWidgetEntry
    
    var body: some View {
        if let data = entry.widgetData {
            HStack(spacing: 4) {
                Image(systemName: data.isRunning ? "checkmark.circle.fill" : "xmark.circle.fill")
                Text(data.formattedBalance)
                Text(data.formattedDailyPnL)
                    .foregroundColor(data.isInProfit ? .green : .red)
            }
        } else {
            Text("Eden Bot - No Data")
        }
    }
}

// MARK: - Helper Views
struct StatRow: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased())
                .font(.system(size: 8, weight: .medium))
                .foregroundColor(.gray)
            Text(value)
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(.white)
        }
    }
}

struct MetricBox: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 6) {
            Rectangle()
                .fill(color)
                .frame(width: 3, height: 30)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(label.uppercased())
                    .font(.system(size: 8, weight: .medium))
                    .foregroundColor(.gray)
                Text(value)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.white)
            }
        }
    }
}

struct PlaceholderView: View {
    let text: String
    
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.1, green: 0.1, blue: 0.15), Color(red: 0.05, green: 0.05, blue: 0.1)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            
            VStack(spacing: 8) {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .font(.system(size: 32))
                    .foregroundColor(.gray)
                Text(text)
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }
        }
    }
}
