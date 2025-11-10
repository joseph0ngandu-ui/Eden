//
//  RecentTradesView.swift
//  Eden
//
//  Recent trades list component
//

import SwiftUI

struct RecentTradesView: View {
    @EnvironmentObject var botManager: BotManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recent Trades")
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.white)
            
            ForEach(botManager.recentTrades) { trade in
                TradeRow(trade: trade)
            }
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
    }
}

#Preview {
    RecentTradesView()
        .environmentObject(BotManager())
        .padding()
        .background(Color.black)
}
