//
//  BotManager.swift
//  Eden
//
//  State management for the bot
//

import Foundation
import Combine

class BotManager: ObservableObject {
    // Published properties
    @Published var balance: Double = 347.82
    @Published var initialBalance: Double = 100
    @Published var peakBalance: Double = 389.45
    @Published var dailyPnL: Double = 23.45
    @Published var totalReturn: Double = 247.82
    @Published var isRunning: Bool = true
    @Published var winRate: Double = 68.5
    @Published var riskTier: String = "AGGRESSIVE"
    @Published var riskPerTrade: Double = 5.0
    @Published var activePositions: Int = 2
    @Published var totalTrades: Int = 127
    @Published var profitFactor: Double = 2.34
    @Published var currentDrawdown: Double = 10.7
    
    @Published var equityData: [EquityPoint] = [
        EquityPoint(time: "9:00", value: 324.50),
        EquityPoint(time: "10:00", value: 329.20),
        EquityPoint(time: "11:00", value: 334.80),
        EquityPoint(time: "12:00", value: 338.90),
        EquityPoint(time: "13:00", value: 342.10),
        EquityPoint(time: "14:00", value: 347.82)
    ]
    
    @Published var recentTrades: [Trade] = [
        Trade(symbol: "XAUUSD", pnl: 18.45, time: "13:42", rValue: 2.1),
        Trade(symbol: "V100", pnl: 12.30, time: "13:15", rValue: 1.8),
        Trade(symbol: "BOOM500", pnl: -5.20, time: "12:48", rValue: -0.6),
        Trade(symbol: "V75", pnl: 22.10, time: "12:20", rValue: 2.5)
    ]
    
    @Published var positions: [Position] = [
        Position(symbol: "XAUUSD", direction: "LONG", entry: 1950.34, current: 1958.20, pnl: 15.72, confidence: 0.94, bars: 3),
        Position(symbol: "V75", direction: "SHORT", entry: 245.80, current: 243.10, pnl: 8.10, confidence: 0.87, bars: 5)
    ]
    
    private var timer: Timer?
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        startRealtimeUpdates()
        // Uncomment to fetch real data from API
        // fetchBotStatus()
    }
    
    // MARK: - Bot Controls
    func toggleBot() {
        isRunning.toggle()
        // TODO: Call Eden API to start/stop bot
        APIService.shared.controlBot(command: isRunning ? "start" : "stop") { result in
            switch result {
            case .success(let message):
                print("✓ Bot control: \(message)")
            case .failure(let error):
                print("✗ Bot control error: \(error)")
            }
        }
    }
    
    func saveSettings() {
        // TODO: Save settings to UserDefaults and update API
        print("Settings saved")
    }
    
    // MARK: - Real-time Updates
    func startRealtimeUpdates() {
        timer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.updateData()
        }
    }
    
    private func updateData() {
        // Simulate real-time updates
        balance += Double.random(in: -2...2)
        dailyPnL += Double.random(in: -0.5...0.5)
        totalReturn = ((balance / initialBalance) - 1) * 100
        
        // Update equity data
        let newPoint = EquityPoint(
            time: DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .short),
            value: balance
        )
        equityData.append(newPoint)
        if equityData.count > 6 {
            equityData.removeFirst()
        }
        
        // Update positions
        if !positions.isEmpty {
            positions = positions.map { position in
                let change = Double.random(in: -2...3)
                return Position(
                    symbol: position.symbol,
                    direction: position.direction,
                    entry: position.entry,
                    current: position.current + change,
                    pnl: position.pnl + change * 0.5,
                    confidence: position.confidence,
                    bars: position.bars + 1
                )
            }
        }
        
        // Save data for widgets
        saveDataForWidgets()
    }
    
    // MARK: - API Data Fetching
    func fetchBotStatus() {
        APIService.shared.fetchBotStatus { [weak self] result in
            switch result {
            case .success(let status):
                self?.updateWithStatus(status)
            case .failure(let error):
                print("Error fetching bot status: \(error)")
            }
        }
    }
    
    func fetchPositions() {
        APIService.shared.fetchPositions { [weak self] result in
            switch result {
            case .success(let positions):
                self?.positions = positions
                self?.activePositions = positions.count
            case .failure(let error):
                print("Error fetching positions: \(error)")
            }
        }
    }
    
    func fetchTrades() {
        APIService.shared.fetchTrades { [weak self] result in
            switch result {
            case .success(let trades):
                self?.recentTrades = trades
            case .failure(let error):
                print("Error fetching trades: \(error)")
            }
        }
    }
    
    private func updateWithStatus(_ status: BotStatus) {
        isRunning = status.isRunning
        balance = status.balance
        dailyPnL = status.dailyPnL
        activePositions = status.activePositions
        winRate = status.winRate
        riskTier = status.riskTier
        
        if let totalTrades = status.totalTrades {
            self.totalTrades = totalTrades
        }
        if let profitFactor = status.profitFactor {
            self.profitFactor = profitFactor
        }
        if let peakBalance = status.peakBalance {
            self.peakBalance = peakBalance
        }
        if let currentDrawdown = status.currentDrawdown {
            self.currentDrawdown = currentDrawdown
        }
        
        // Save data for widgets
        saveDataForWidgets()
    }
    
    // MARK: - Widget Data Management
    private func saveDataForWidgets() {
        SharedDataService.shared.saveBotStatus(
            isRunning: isRunning,
            balance: balance,
            dailyPnL: dailyPnL,
            activePositions: activePositions,
            winRate: winRate,
            totalTrades: totalTrades,
            profitFactor: profitFactor,
            riskTier: riskTier
        )
    }
    
    deinit {
        timer?.invalidate()
    }
}
