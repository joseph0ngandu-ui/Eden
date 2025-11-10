//
//  AnalyticsView.swift
//  Eden
//
//  Performance metrics and analytics screen
//

import SwiftUI

struct AnalyticsView: View {
    @EnvironmentObject var botManager: BotManager
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 20) {
                Text("Performance Metrics")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
                
                VStack(spacing: 16) {
                    MetricRow(label: "Peak Balance", value: "$\(botManager.peakBalance, specifier: "%.2f")")
                    MetricRow(label: "Current Drawdown", value: "\(botManager.currentDrawdown, specifier: "%.1f")%", valueColor: .red)
                    MetricRow(label: "Total Trades", value: "\(botManager.totalTrades)")
                    MetricRow(label: "Profit Factor", value: "\(botManager.profitFactor, specifier: "%.2f")", valueColor: .green)
                    MetricRow(label: "Win Rate", value: "\(botManager.winRate, specifier: "%.1f")%", valueColor: .green)
                    MetricRow(label: "Risk Tier", value: botManager.riskTier, valueColor: .purple)
                }
                .padding(20)
                .background(
                    ZStack {
                        RoundedRectangle(cornerRadius: 24)
                            .fill(Color.gray.opacity(0.15))
                        
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color.white.opacity(0.1), lineWidth: 1)
                    }
                )
                .padding(.horizontal)
                
                Spacer(minLength: 100)
            }
            .padding(.top, 20)
        }
    }
}

#Preview {
    AnalyticsView()
        .environmentObject(BotManager())
        .background(Color.black)
}
