import SwiftUI

struct TestingView: View {
    @StateObject private var accountService = MT5AccountService.shared
    @State private var isResetting = false
    @State private var showConfirmation = false
    @State private var message: String?
    @State private var isError = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Text("Testing & Simulation")
                    .font(.title)
                    .fontWeight(.bold)
                Spacer()
            }
            .padding()
            
            Divider()
            
            ScrollView {
                VStack(spacing: 24) {
                    // Paper Trading Section
                    GroupBox("Paper Trading") {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Manage your paper trading environment. Resetting will clear all paper positions, history, and restore balance to default.")
                                .foregroundColor(.secondary)
                            
                            Button(role: .destructive) {
                                showConfirmation = true
                            } label: {
                                Label("Reset Paper Account", systemImage: "arrow.counterclockwise")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.large)
                            .disabled(isResetting)
                        }
                        .padding()
                    }
                    
                    // Simulation Section (Placeholder)
                    GroupBox("Market Simulation") {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Simulate market conditions to test strategy behavior.")
                                .foregroundColor(.secondary)
                            
                            Toggle("Simulate High Volatility", isOn: .constant(false))
                                .disabled(true)
                            
                            Toggle("Simulate Network Latency", isOn: .constant(false))
                                .disabled(true)
                                
                            Text("Simulation features coming soon.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                    }
                }
                .padding()
            }
            
            if let message = message {
                HStack {
                    Image(systemName: isError ? "exclamationmark.triangle" : "checkmark.circle")
                    Text(message)
                    Spacer()
                    Button("Dismiss") {
                        self.message = nil
                    }
                }
                .padding()
                .background(isError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
            }
        }
        .alert("Reset Paper Account?", isPresented: $showConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Reset", role: .destructive) {
                resetAccount()
            }
        } message: {
            Text("This action cannot be undone. All paper trading data will be lost.")
        }
    }
    
    private func resetAccount() {
        isResetting = true
        message = nil
        
        Task {
            do {
                try await accountService.resetPaperAccount()
                await MainActor.run {
                    isResetting = false
                    isError = false
                    message = "Paper account reset successfully."
                }
            } catch {
                await MainActor.run {
                    isResetting = false
                    isError = true
                    message = "Failed to reset account: \(error.localizedDescription)"
                }
            }
        }
    }
}

#Preview {
    TestingView()
}
