//
//  HeaderView.swift
//  Eden
//
//  App header with logo, balance, and bot controls
//

import SwiftUI

struct HeaderView: View {
    @EnvironmentObject var botManager: BotManager
    @State private var balanceVisible = true
    
    var body: some View {
        VStack(spacing: 0) {
            // Top Bar
            HStack {
                // Logo
                HStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(LinearGradient(
                                colors: [.purple, .blue],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ))
                            .frame(width: 50, height: 50)
                            .blur(radius: 8)
                        
                        Circle()
                            .fill(LinearGradient(
                                colors: [.purple.opacity(0.9), .blue.opacity(0.9)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ))
                            .frame(width: 50, height: 50)
                        
                        Image(systemName: "bolt.fill")
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)
                    }
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Eden")
                            .font(.system(size: 28, weight: .bold))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [.white, .purple.opacity(0.8), .blue.opacity(0.8)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                        
                        Text("AI Trading System")
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                    }
                }
                
                Spacer()
                
                // Bot Control Button
                Button(action: { botManager.toggleBot() }) {
                    HStack(spacing: 8) {
                        Image(systemName: botManager.isRunning ? "pause.circle.fill" : "play.circle.fill")
                            .font(.system(size: 20))
                        
                        Text(botManager.isRunning ? "Active" : "Paused")
                            .font(.system(size: 16, weight: .semibold))
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .background(
                        LinearGradient(
                            colors: botManager.isRunning ? [.green, .green.opacity(0.8)] : [.red, .red.opacity(0.8)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(16)
                    .shadow(color: botManager.isRunning ? .green.opacity(0.5) : .red.opacity(0.5), radius: 10)
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 20)
            .background(
                ZStack {
                    RoundedRectangle(cornerRadius: 24)
                        .fill(LinearGradient(
                            colors: [Color.gray.opacity(0.2), Color.gray.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ))
                    
                    RoundedRectangle(cornerRadius: 24)
                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                }
            )
            .overlay(
                RoundedRectangle(cornerRadius: 24)
                    .fill(
                        LinearGradient(
                            colors: [.purple.opacity(0.1), .blue.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .blur(radius: 20)
                    .opacity(0.5)
            )
            
            // Balance Card
            VStack(spacing: 12) {
                HStack {
                    Text("Account Balance")
                        .font(.system(size: 14))
                        .foregroundColor(.gray)
                    
                    Spacer()
                    
                    Button(action: { balanceVisible.toggle() }) {
                        Image(systemName: balanceVisible ? "eye.fill" : "eye.slash.fill")
                            .font(.system(size: 14))
                            .foregroundColor(.gray)
                    }
                }
                
                Text(balanceVisible ? "$\(botManager.balance, specifier: "%.2f")" : "••••••")
                    .font(.system(size: 48, weight: .bold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.green, .green.opacity(0.8)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                
                HStack(spacing: 20) {
                    HStack(spacing: 4) {
                        Image(systemName: botManager.dailyPnL >= 0 ? "arrow.up.right" : "arrow.down.right")
                            .font(.system(size: 12))
                        
                        Text("\(botManager.dailyPnL >= 0 ? "+" : "")\(botManager.dailyPnL, specifier: "%.2f")")
                            .font(.system(size: 14, weight: .semibold))
                        
                        Text("today")
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                    }
                    .foregroundColor(botManager.dailyPnL >= 0 ? .green : .red)
                    
                    Text("•")
                        .foregroundColor(.gray)
                    
                    HStack(spacing: 4) {
                        Text("\(botManager.totalReturn, specifier: "%.1f")%")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(.purple)
                        
                        Text("total")
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                    }
                }
            }
            .padding(24)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color.gray.opacity(0.15))
            )
            .padding(.horizontal, 20)
            .padding(.top, 12)
        }
    }
}

#Preview {
    HeaderView()
        .environmentObject(BotManager())
        .background(Color.black)
}
