//
//  ContentView.swift
//  Eden
//
//  Main container view with tab navigation
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var botManager: BotManager
    @State private var selectedTab = 0
    
    var body: some View {
        ZStack {
            // Background
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Header
                HeaderView()
                    .padding(.horizontal)
                    .padding(.top, 20)
                
                // Tab Content
                TabView(selection: $selectedTab) {
                    OverviewView()
                        .tag(0)
                    
                    PositionsView()
                        .tag(1)
                    
                    AnalyticsView()
                        .tag(2)
                    
                    SettingsView()
                        .tag(3)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                
                // Custom Tab Bar
                CustomTabBar(selectedTab: $selectedTab)
                    .padding(.horizontal)
                    .padding(.bottom, 10)
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(BotManager())
}
