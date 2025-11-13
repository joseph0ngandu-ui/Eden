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
    
    // MetaTrader Account Settings
    @State private var mt5AccountNumber = ""
    @State private var mt5AccountName = ""
    @State private var mt5Broker = ""
    @State private var mt5Server = ""
    @State private var mt5Password = ""
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 20) {
                Text("Settings")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
                
                // API Configuration Section
                VStack(alignment: .leading, spacing: 12) {
                    Text("API Configuration")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.white)
                    
                    SettingField(label: "Webhook URL", text: $webhookURL)
                    SettingField(label: "API Key", text: $apiKey, isSecure: true)
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
                
                // MetaTrader Account Section
                VStack(alignment: .leading, spacing: 12) {
                    Text("MetaTrader 5 Account")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.white)
                    
                    Text("Configure the MT5 account Eden is trading on")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)
                    
                    SettingField(label: "Account Number", text: $mt5AccountNumber)
                        .keyboardType(.numberPad)
                    
                    SettingField(label: "Account Name", text: $mt5AccountName)
                    
                    SettingField(label: "Broker", text: $mt5Broker)
                    
                    SettingField(label: "Server", text: $mt5Server)
                    
                    SettingField(label: "Password", text: $mt5Password, isSecure: true)
                    
                    // Display account info if available
                    if !mt5AccountNumber.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Trading Account:")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.gray)
                            
                            HStack {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("Account \(mt5AccountNumber) - \(mt5AccountName)")
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(.white)
                            }
                            
                            Text("\(mt5Broker) - \(mt5Server)")
                                .font(.system(size: 12))
                                .foregroundColor(.gray)
                        }
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.green.opacity(0.1))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.green.opacity(0.3), lineWidth: 1)
                        )
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
                
                // Save Button
                Button(action: { 
                    saveAllSettings()
                }) {
                    Text("Save All Settings")
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
                .padding(.horizontal)
                
                Spacer(minLength: 100)
            }
            .padding(.top, 20)
        }
        .onAppear {
            loadSettings()
        }
    }
    
    // MARK: - Methods
    private func loadSettings() {
        // Load from UserDefaults
        webhookURL = UserDefaults.standard.string(forKey: "webhookURL") ?? "https://your-n8n.com/webhook/eden"
        apiKey = UserDefaults.standard.string(forKey: "apiKey") ?? ""
        
        mt5AccountNumber = UserDefaults.standard.string(forKey: "mt5AccountNumber") ?? ""
        mt5AccountName = UserDefaults.standard.string(forKey: "mt5AccountName") ?? ""
        mt5Broker = UserDefaults.standard.string(forKey: "mt5Broker") ?? ""
        mt5Server = UserDefaults.standard.string(forKey: "mt5Server") ?? ""
        mt5Password = UserDefaults.standard.string(forKey: "mt5Password") ?? ""
    }
    
    private func saveAllSettings() {
        // Save API settings
        UserDefaults.standard.set(webhookURL, forKey: "webhookURL")
        UserDefaults.standard.set(apiKey, forKey: "apiKey")
        
        // Save MT5 account settings
        UserDefaults.standard.set(mt5AccountNumber, forKey: "mt5AccountNumber")
        UserDefaults.standard.set(mt5AccountName, forKey: "mt5AccountName")
        UserDefaults.standard.set(mt5Broker, forKey: "mt5Broker")
        UserDefaults.standard.set(mt5Server, forKey: "mt5Server")
        UserDefaults.standard.set(mt5Password, forKey: "mt5Password")
        
        // Call bot manager to sync settings
        botManager.saveSettings()
        
        // Show success feedback (you could add haptic feedback here)
        print("✅ All settings saved successfully")
        print("MT5 Account: \(mt5AccountNumber) - \(mt5AccountName)")
        print("Broker: \(mt5Broker) | Server: \(mt5Server)")
    }
}

#Preview {
    SettingsView()
        .environmentObject(BotManager())
        .background(Color.black)
}
