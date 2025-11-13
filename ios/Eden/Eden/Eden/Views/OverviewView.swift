//
//  OverviewView.swift
//  Eden
//
//  Main dashboard with stats, equity curve, and recent trades
//

import SwiftUI

struct OverviewView: View {
    @EnvironmentObject var botManager: BotManager
    @State private var mt5AccountNumber = ""
    @State private var mt5AccountName = ""
    @State private var mt5Broker = ""
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 20) {
                // MT5 Account Info Card
                if !mt5AccountNumber.isEmpty {
                    HStack(spacing: 12) {
                        Image(systemName: "building.columns.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.green)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Trading Account")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.gray)
                            
                            Text("\(mt5AccountNumber) - \(mt5AccountName)")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(.white)
                            
                            Text(mt5Broker)
                                .font(.system(size: 13))
                                .foregroundColor(.gray)
                        }
                        
                        Spacer()
                        
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                    .padding(16)
                    .background(
                        ZStack {
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color.gray.opacity(0.15))
                            
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.green.opacity(0.3), lineWidth: 1)
                        }
                    )
                    .padding(.horizontal)
                }
                
                // Stats Grid
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                    StatCard(
                        icon: "target",
                        label: "Win Rate",
                        value: String(format: "%.1f%%", botManager.winRate),
                        trend: 2.3,
                        color: .green
                    )
                    
                    StatCard(
                        icon: "shield.fill",
                        label: "Risk Tier",
                        value: botManager.riskTier,
                        subtext: String(format: "%.1f%% per trade", botManager.riskPerTrade),
                        color: .purple
                    )
                    
                    StatCard(
                        icon: "chart.line.uptrend.xyaxis",
                        label: "Active Trades",
                        value: "\(botManager.activePositions)",
                        subtext: "\(botManager.totalTrades) total",
                        color: .orange
                    )
                    
                    StatCard(
                        icon: "chart.bar.fill",
                        label: "Profit Factor",
                        value: String(format: "%.2f", botManager.profitFactor),
                        trend: 5.1,
                        color: .cyan
                    )
                }
                .padding(.horizontal)
                
                // Equity Curve
                EquityCurveView()
                    .frame(height: 250)
                    .padding(.horizontal)
                
                // Recent Trades
                RecentTradesView()
                    .padding(.horizontal)
                
                Spacer(minLength: 100)
            }
            .padding(.top, 20)
        }
        .onAppear {
            loadMT5AccountInfo()
        }
    }
    
    // MARK: - Methods
    private func loadMT5AccountInfo() {
        mt5AccountNumber = UserDefaults.standard.string(forKey: "mt5AccountNumber") ?? ""
        mt5AccountName = UserDefaults.standard.string(forKey: "mt5AccountName") ?? ""
        mt5Broker = UserDefaults.standard.string(forKey: "mt5Broker") ?? ""
    }
}

#Preview {
    OverviewView()
        .environmentObject(BotManager())
        .background(Color.black)
}
