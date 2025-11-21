import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var authService: AuthService
    @StateObject private var notificationManager = NotificationManager.shared
    @State private var apiURL: String = APIService.shared.baseURL
    @State private var showingResetConfirmation = false
    @AppStorage("darkMode") private var darkMode: Bool = false
    @AppStorage("autoConnect") private var autoConnect: Bool = true
    @AppStorage("notificationsEnabled") private var notificationsEnabled: Bool = true

    var body: some View {
        Form {
            // Connection Settings
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    TextField("API Base URL", text: $apiURL)
                        .onChange(of: apiURL) { _, newValue in
                            APIService.shared.baseURL = newValue
                        }

                    Text("Example: https://edenbot.duckdns.org:8443")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Toggle("Auto-connect on launch", isOn: $autoConnect)

                Button {
                    testConnection()
                } label: {
                    Label("Test Connection", systemImage: "network")
                }
            } header: {
                Label("Connection", systemImage: "network")
            }

            // Notifications
            Section {
                Toggle("Enable Notifications", isOn: $notificationsEnabled)
                    .onChange(of: notificationsEnabled) { _, newValue in
                        if newValue && !notificationManager.hasPermission {
                            notificationManager.requestPermissions()
                        }
                    }

                if notificationsEnabled {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(
                                systemName: notificationManager.hasPermission
                                    ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                            )
                            .foregroundStyle(notificationManager.hasPermission ? .green : .orange)

                            Text(
                                notificationManager.hasPermission
                                    ? "Notifications allowed" : "Notifications blocked"
                            )
                            .font(.subheadline)

                            Spacer()

                            if !notificationManager.hasPermission {
                                Button("Grant Permission") {
                                    notificationManager.requestPermissions()
                                }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.small)
                            }
                        }

                        Text("Get notified about trades, position changes, and connection status")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            } header: {
                Label("Notifications", systemImage: "bell.badge")
            }

            // Appearance
            Section {
                Toggle("Dark Mode", isOn: $darkMode)
                    .onChange(of: darkMode) { _, newValue in
                        // Apply dark mode preference
                        if let window = NSApplication.shared.windows.first {
                            window.appearance = NSAppearance(named: newValue ? .darkAqua : .aqua)
                        }
                    }
            } header: {
                Label("Appearance", systemImage: "paintbrush")
            }

            // Account
            Section {
                if let user = authService.currentUser {
                    HStack {
                        Label("Email", systemImage: "person.circle")
                        Spacer()
                        Text(user)
                            .foregroundColor(.secondary)
                    }
                }

                Button(role: .destructive) {
                    authService.logout()
                } label: {
                    Label("Sign Out", systemImage: "rectangle.portrait.and.arrow.right")
                }
            } header: {
                Label("Account", systemImage: "person")
            }

            // Advanced
            Section {
                Button(role: .destructive) {
                    showingResetConfirmation = true
                } label: {
                    Label("Reset All Settings", systemImage: "arrow.counterclockwise")
                }
            } header: {
                Label("Advanced", systemImage: "gearshape.2")
            }

            // About
            Section {
                LabeledContent("Version", value: "1.0.0")
                LabeledContent("Build", value: "1")

                Link(destination: URL(string: "https://github.com/yourusername/eden")!) {
                    Label("View on GitHub", systemImage: "link")
                }
            } header: {
                Label("About", systemImage: "info.circle")
            }
        }
        .formStyle(.grouped)
        .frame(width: 550, height: 600)
        .alert("Reset Settings", isPresented: $showingResetConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Reset", role: .destructive) {
                resetSettings()
            }
        } message: {
            Text(
                "This will reset all settings to their default values. This action cannot be undone."
            )
        }
    }

    private func testConnection() {
        // TODO: Implement connection test
        print("Testing connection to: \(apiURL)")
    }

    private func resetSettings() {
        apiURL = "https://edenbot.duckdns.org:8443"
        APIService.shared.baseURL = apiURL
        autoConnect = true
        notificationsEnabled = true
        darkMode = false
    }
}

#Preview {
    SettingsView()
        .environmentObject(AuthService.shared)
}
