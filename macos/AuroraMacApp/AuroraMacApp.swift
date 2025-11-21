import SwiftUI

@main
struct AuroraMacApp: App {
    @StateObject private var authService = AuthService.shared
    @StateObject private var strategyViewModel = StrategyViewModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authService)
                .environmentObject(strategyViewModel)
                .frame(minWidth: 1000, minHeight: 700)
        }
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
        
        Settings {
            SettingsView()
                .environmentObject(authService)
        }
    }
}
