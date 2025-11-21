import SwiftUI

struct ErrorPresenter {
    enum ErrorType {
        case network
        case authentication
        case validation
        case unknown

        var icon: String {
            switch self {
            case .network: return "wifi.exclamationmark"
            case .authentication: return "lock.trianglebadge.exclamationmark"
            case .validation: return "exclamationmark.triangle"
            case .unknown: return "exclamationmark.circle"
            }
        }

        var color: Color {
            switch self {
            case .network: return .orange
            case .authentication: return .red
            case .validation: return .yellow
            case .unknown: return .gray
            }
        }
    }

    static func userFriendlyMessage(for error: Error) -> String {
        if let apiError = error as? APIService.APIError {
            return message(for: apiError)
        }

        if let authError = error as? AuthService.KeychainError {
            return message(for: authError)
        }

        // Check for network-related errors
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            switch nsError.code {
            case NSURLErrorNotConnectedToInternet:
                return "No internet connection. Please check your network settings."
            case NSURLErrorTimedOut:
                return "Request timed out. Please try again."
            case NSURLErrorCannotFindHost, NSURLErrorCannotConnectToHost:
                return "Cannot connect to server. Please check the API URL in settings."
            case NSURLErrorSecureConnectionFailed:
                return "Secure connection failed. Please verify the server certificate."
            default:
                return "Network error: \(error.localizedDescription)"
            }
        }

        return error.localizedDescription
    }

    static func message(for error: APIService.APIError) -> String {
        switch error {
        case .authenticationFailed:
            return "Login failed. Please check your email and password."
        case .networkError:
            return "Network error. Please check your connection and try again."
        case .uploadFailed:
            return "Failed to upload strategy. Please try again."
        case .invalidResponse:
            return "Invalid response from server. Please contact support."
        }
    }

    static func message(for error: AuthService.KeychainError) -> String {
        switch error {
        case .saveFailed:
            return "Failed to save credentials securely. Please try again."
        case .loadFailed:
            return "Failed to load saved credentials. You may need to login again."
        }
    }

    static func errorType(for error: Error) -> ErrorType {
        if error is APIService.APIError {
            return .network
        }

        if error is AuthService.KeychainError {
            return .authentication
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            return .network
        }

        return .unknown
    }
}

// SwiftUI view for displaying errors
struct ErrorBanner: View {
    let message: String
    let type: ErrorPresenter.ErrorType
    var onDismiss: (() -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: type.icon)
                .font(.title3)
                .foregroundStyle(type.color)

            Text(message)
                .font(.subheadline)

            Spacer()

            if let onDismiss = onDismiss {
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(type.color.opacity(0.15))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(type.color.opacity(0.5), lineWidth: 1)
        )
    }
}

#Preview {
    VStack(spacing: 16) {
        ErrorBanner(
            message: "Network connection lost. Retrying...",
            type: .network
        )

        ErrorBanner(
            message: "Authentication failed. Please check your credentials.",
            type: .authentication,
            onDismiss: {}
        )

        ErrorBanner(
            message: "Invalid input. Please check your data.",
            type: .validation
        )
    }
    .padding()
    .frame(width: 500)
}
