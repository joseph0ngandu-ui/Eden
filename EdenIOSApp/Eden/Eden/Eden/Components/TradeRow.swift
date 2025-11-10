//
//  TradeRow.swift
//  Eden
//
//  Individual trade row component
//

import SwiftUI

struct TradeRow: View {
    let trade: Trade
    
    var body: some View {
        HStack {
            Circle()
                .fill(trade.pnl >= 0 ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(trade.symbol)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.white)
                
                Text(trade.time)
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(trade.pnl >= 0 ? "+" : "")\(trade.pnl, specifier: "%.2f")")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(trade.pnl >= 0 ? .green : .red)
                
                Text("\(trade.rValue, specifier: "%.1f")R")
                    .font(.system(size: 11))
                    .foregroundColor(.gray)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.gray.opacity(0.1))
        )
    }
}

#Preview {
    TradeRow(trade: Trade(symbol: "XAUUSD", pnl: 18.45, time: "13:42", rValue: 2.1))
        .padding()
        .background(Color.black)
}
