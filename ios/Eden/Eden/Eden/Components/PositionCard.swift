//
//  PositionCard.swift
//  Eden
//
//  Active position card component
//

import SwiftUI

struct PositionCard: View {
    let position: Position
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text(position.symbol)
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.white)
                
                Text(position.direction)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(position.direction == "LONG" ? .green : .red)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(position.direction == "LONG" ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                    )
                
                Spacer()
                
                Text("\(position.pnl >= 0 ? "+" : "")\(position.pnl, specifier: "%.2f")")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(position.pnl >= 0 ? .green : .red)
            }
            
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Entry")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                    Text("\(position.entry, specifier: "%.2f")")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.white)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Current")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                    Text("\(position.current, specifier: "%.2f")")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.white)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Bars")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                    Text("\(position.bars)/12")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.white)
                }
                
                Spacer()
            }
            
            HStack {
                Image(systemName: "sparkles")
                    .font(.system(size: 12))
                    .foregroundColor(.purple)
                
                Text("Confidence")
                    .font(.system(size: 13))
                    .foregroundColor(.gray)
                
                Spacer()
                
                Text("\(Int(position.confidence * 100))%")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white)
            }
            
            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.gray.opacity(0.2))
                    
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [.purple, .blue],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * CGFloat(position.confidence))
                }
            }
            .frame(height: 8)
        }
        .padding(20)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color.gray.opacity(0.15))
                
                RoundedRectangle(cornerRadius: 20)
                    .stroke(Color.white.opacity(0.1), lineWidth: 1)
            }
        )
    }
}

#Preview {
    PositionCard(position: Position(symbol: "XAUUSD", direction: "LONG", entry: 1950.34, current: 1958.20, pnl: 15.72, confidence: 0.94, bars: 3))
        .padding()
        .background(Color.black)
}
