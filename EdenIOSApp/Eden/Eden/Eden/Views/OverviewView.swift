//
//  OverviewView.swift
//  Eden
//
//  Main dashboard with stats, equity curve, and recent trades
//

import SwiftUI

struct OverviewView: View {
    @EnvironmentObject var botManager: BotManager
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 20) {
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
    }
}

#Preview {
    OverviewView()
        .environmentObject(BotManager())
        .background(Color.black)
}
