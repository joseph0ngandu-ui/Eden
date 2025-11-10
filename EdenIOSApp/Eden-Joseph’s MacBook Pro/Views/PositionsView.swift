//
//  PositionsView.swift
//  Eden
//
//  Active positions monitoring screen
//

import SwiftUI

struct PositionsView: View {
    @EnvironmentObject var botManager: BotManager
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 16) {
                HStack {
                    Text("Active Positions")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    
                    Spacer()
                    
                    Text("\(botManager.activePositions) open")
                        .font(.system(size: 14))
                        .foregroundColor(.gray)
                }
                .padding(.horizontal)
                
                ForEach(botManager.positions) { position in
                    PositionCard(position: position)
                        .padding(.horizontal)
                }
                
                Spacer(minLength: 100)
            }
            .padding(.top, 20)
        }
    }
}

#Preview {
    PositionsView()
        .environmentObject(BotManager())
        .background(Color.black)
}
