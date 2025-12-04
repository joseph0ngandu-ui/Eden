import SwiftUI
import Combine

struct LoginView: View {
    @EnvironmentObject var authService: AuthService
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 60))
                .foregroundStyle(.blue)

            Text("Aurora Mac")
                .font(.title)
                .fontWeight(.bold)

            Text("ML Strategy Trainer")
                .font(.subheadline)
                .foregroundColor(.secondary)

            Divider()
                .padding(.vertical)

            VStack(alignment: .leading, spacing: 12) {
                Text("Email")
                    .font(.caption)
                    .foregroundColor(.secondary)

                TextField("you@example.com", text: $email)
                    .textFieldStyle(.roundedBorder)

                Text("Password")
                    .font(.caption)
                    .foregroundColor(.secondary)

                SecureField("••••••••", text: $password)
                    .textFieldStyle(.roundedBorder)
            }
            .padding(.horizontal)

            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
            }

            Button {
                Task {
                    await performLogin()
                }
            } label: {
                if isLoading {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Text("Sign In")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(email.isEmpty || password.isEmpty || isLoading)
        }
        .frame(width: 400, height: 500)
        .padding()
    }

    private func performLogin() async {
        isLoading = true
        errorMessage = nil

        do {
            try await authService.login(email: email, password: password)
        } catch {
            errorMessage = "Login failed: \(error.localizedDescription)"
        }

        isLoading = false
    }
}

#Preview {
    LoginView()
        .environmentObject(AuthService.shared)
}
