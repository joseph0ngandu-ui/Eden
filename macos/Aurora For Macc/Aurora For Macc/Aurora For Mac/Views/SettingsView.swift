import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var authService: AuthService
    @State private var apiURL: String = APIService.shared.baseURL

    var body: some View {
        Form {
            Section("Connection") {
                TextField("API Base URL", text: $apiURL)
                    .onChange(of: apiURL) { _, newValue in
                        APIService.shared.baseURL = newValue
                    }

                Text("Example: https://edenbot.duckdns.org:8443")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section("Account") {
                if let user = authService.currentUser {
                    LabeledContent("Logged in as", value: user)
                }

                Button("Sign Out") {
                    authService.logout()
                }
                .foregroundColor(.red)
            }
        }
        .formStyle(.grouped)
        .frame(width: 450, height: 300)
    }
}

#Preview {
    SettingsView()
        .environmentObject(AuthService.shared)
}
