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
                        value: "\(botManager.winRate, specifier: "%.1f")%",
                        trend: 2.3,
                        color: .green
                    )
                    
                    StatCard(
                        icon: "shield.fill",
                        label: "Risk Tier",
                        value: botManager.riskTier,
                        subtext: "\(botManager.riskPerTrade, specifier: "%.1f")% per trade",
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
                        value: "\(botManager.profitFactor, specifier: "%.2f")",
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
