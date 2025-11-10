//
//  EquityCurveView.swift
//  Eden
//
//  Equity curve chart component
//

import SwiftUI

struct EquityCurveView: View {
    @EnvironmentObject var botManager: BotManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Equity Curve")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.white)
                
                Spacer()
                
                Text("Last 6 hours")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }
            
            // Simple line chart representation
            GeometryReader { geometry in
                Path { path in
                    let points = botManager.equityData
                    guard !points.isEmpty else { return }
                    
                    let maxValue = points.map { $0.value }.max() ?? 1
                    let minValue = points.map { $0.value }.min() ?? 0
                    let range = maxValue - minValue
                    
                    let stepX = geometry.size.width / CGFloat(points.count - 1)
                    
                    for (index, point) in points.enumerated() {
                        let x = CGFloat(index) * stepX
                        let y = geometry.size.height - (CGFloat(point.value - minValue) / CGFloat(range)) * geometry.size.height
                        
                        if index == 0 {
                            path.move(to: CGPoint(x: x, y: y))
                        } else {
                            path.addLine(to: CGPoint(x: x, y: y))
                        }
                    }
                }
                .stroke(
                    LinearGradient(
                        colors: [.purple, .blue],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round)
                )
                .shadow(color: .purple.opacity(0.5), radius: 8)
            }
            .frame(height: 150)
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
    EquityCurveView()
        .environmentObject(BotManager())
        .padding()
        .background(Color.black)
}
