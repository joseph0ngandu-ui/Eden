//
//  EdenApp.swift
//  Eden
//
//  Created by Eden Trading System
//

import SwiftUI

@main
struct EdenApp: App {
    @StateObject private var botManager = BotManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(botManager)
                .onAppear { WebSocketService.shared.connect() }
                .preferredColorScheme(.dark)
        }
    }
}
