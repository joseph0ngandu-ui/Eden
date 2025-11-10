//
//  SettingsView.swift
//  Eden
//
//  Bot configuration and settings screen
//

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var botManager: BotManager
    @State private var webhookURL = "https://your-n8n.com/webhook/eden"
    @State private var apiKey = ""
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 20) {
                Text("Settings")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
                
                VStack(spacing: 16) {
                    SettingField(label: "Webhook URL", text: $webhookURL)
                    SettingField(label: "API Key", text: $apiKey, isSecure: true)
                    
                    Button(action: { botManager.saveSettings() }) {
                        Text("Save Settings")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(
                                LinearGradient(
                                    colors: [.purple, .blue],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .cornerRadius(16)
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
                .padding(.horizontal)
                
                Spacer(minLength: 100)
            }
            .padding(.top, 20)
        }
    }
}

#Preview {
    SettingsView()
        .environmentObject(BotManager())
        .background(Color.black)
}
