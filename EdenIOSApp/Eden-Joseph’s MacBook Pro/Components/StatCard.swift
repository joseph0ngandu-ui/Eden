//
//  StatCard.swift
//  Eden
//
//  Reusable stat card component
//

import SwiftUI

struct StatCard: View {
    let icon: String
    let label: String
    let value: String
    var subtext: String? = nil
    var trend: Double? = nil
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                ZStack {
                    Circle()
                        .fill(LinearGradient(
                            colors: [color, color.opacity(0.7)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ))
                        .frame(width: 44, height: 44)
                    
                    Image(systemName: icon)
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(.white)
                }
                
                Spacer()
                
                if let trend = trend {
                    HStack(spacing: 4) {
                        Image(systemName: trend >= 0 ? "arrow.up.right" : "arrow.down.right")
                            .font(.system(size: 10))
                        Text("\(abs(trend), specifier: "%.1f")%")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundColor(trend >= 0 ? .green : .red)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(trend >= 0 ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                    )
                }
            }
            
            Text(label)
                .font(.system(size: 13))
                .foregroundColor(.gray)
            
            Text(value)
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(.white)
            
            if let subtext = subtext {
                Text(subtext)
                    .font(.system(size: 11))
                    .foregroundColor(.gray.opacity(0.8))
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
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
    StatCard(icon: "target", label: "Win Rate", value: "68.5%", trend: 2.3, color: .green)
        .padding()
        .background(Color.black)
}
