import SwiftUI

struct LoginView: View {
    @EnvironmentObject var authService: AuthService
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?
    @State private var showPassword: Bool = false

    var body: some View {
        ZStack {
            // Animated gradient background
            LinearGradient(
                colors: [
                    Color.blue.opacity(0.6),
                    Color.purple.opacity(0.6),
                    Color.pink.opacity(0.4),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 24) {
                Spacer()

                // App icon and title
                VStack(spacing: 16) {
                    ZStack {
                        Circle()
                            .fill(.white.opacity(0.2))
                            .frame(width: 120, height: 120)
                            .blur(radius: 20)

                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 60))
                            .foregroundStyle(.white)
                    }

                    Text("Aurora")
                        .font(.system(size: 42, weight: .bold, design: .rounded))
                        .foregroundColor(.white)

                    Text("ML Strategy Trainer")
                        .font(.title3)
                        .foregroundColor(.white.opacity(0.9))
                }

                Spacer()

                // Login form
                VStack(spacing: 20) {
                    VStack(alignment: .leading, spacing: 16) {
                        // Email field
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Email")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.white.opacity(0.9))

                            TextField("you@example.com", text: $email)
                                .textFieldStyle(GlassTextFieldStyle())
                                .textContentType(.emailAddress)
                                .autocapitalization(.none)
                                .keyboardType(.emailAddress)
                        }

                        // Password field
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Password")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.white.opacity(0.9))

                            HStack {
                                if showPassword {
                                    TextField("••••••••", text: $password)
                                        .textFieldStyle(GlassTextFieldStyle())
                                } else {
                                    SecureField("••••••••", text: $password)
                                        .textFieldStyle(GlassTextFieldStyle())
                                }

                                Button {
                                    showPassword.toggle()
                                } label: {
                                    Image(systemName: showPassword ? "eye.slash.fill" : "eye.fill")
                                        .foregroundColor(.white.opacity(0.7))
                                }
                                .buttonStyle(.plain)
                                .padding(.trailing, 8)
                            }
                        }
                    }

                    // Error message
                    if let error = errorMessage {
                        ErrorBanner(
                            message: error,
                            type: .authentication,
                            onDismiss: { errorMessage = nil }
                        )
                        .transition(.move(edge: .top).combined(with: .opacity))
                    }

                    // Login button
                    Button {
                        Task {
                            await performLogin()
                        }
                    } label: {
                        HStack {
                            if isLoading {
                                ProgressView()
                                    .controlSize(.small)
                                    .tint(.white)
                            } else {
                                Text("Sign In")
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(
                                    .white.opacity(
                                        email.isEmpty || password.isEmpty || isLoading ? 0.3 : 0.9))
                        )
                        .foregroundColor(.blue)
                    }
                    .buttonStyle(.plain)
                    .disabled(email.isEmpty || password.isEmpty || isLoading)
                }
                .padding(32)
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(.ultraThinMaterial)
                        .shadow(color: .black.opacity(0.2), radius: 20, x: 0, y: 10)
                )

                Spacer()
            }
            .padding(40)
        }
        .frame(width: 500, height: 600)
    }

    private func performLogin() async {
        isLoading = true
        errorMessage = nil

        do {
            try await authService.login(email: email, password: password)
        } catch {
            errorMessage = ErrorPresenter.userFriendlyMessage(for: error)
        }

        isLoading = false
    }
}

// Custom text field style with glass morphism
struct GlassTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(.white.opacity(0.2))
            )
            .foregroundColor(.white)
            .font(.body)
    }
}

#Preview {
    LoginView()
        .environmentObject(AuthService.shared)
}
