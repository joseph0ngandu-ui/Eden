import SwiftUI

struct AddMT5AccountView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var viewModel: MT5AccountViewModel
    
    @State private var accountNumber = ""
    @State private var password = ""
    @State private var server = ""
    @State private var broker = ""
    @State private var isSubmitting = false
    @State private var errorMessage: String?
    
    var body: some View {
        NavigationStack {
            Form {
                Section("Account Credentials") {
                    TextField("Account Number", text: $accountNumber)
                        .textContentType(.username)
                    
                    SecureField("Password", text: $password)
                        .textContentType(.password)
                }
                
                Section("Broker Details") {
                    TextField("Server", text: $server)
                        .textContentType(.URL)
                        .autocorrectionDisabled()
                    
                    TextField("Broker Name", text: $broker)
                }
                
                if let error = errorMessage {
                    Section {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Add MT5 Account")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") {
                        submit()
                    }
                    .disabled(accountNumber.isEmpty || password.isEmpty || server.isEmpty || broker.isEmpty || isSubmitting)
                }
            }
            .overlay {
                if isSubmitting {
                    ProgressView()
                }
            }
        }
        .frame(width: 400, height: 400)
    }
    
    private func submit() {
        isSubmitting = true
        errorMessage = nil
        
        let account = MT5AccountCreate(
            accountNumber: accountNumber,
            password: password,
            server: server,
            broker: broker
        )
        
        Task {
            let success = await viewModel.addAccount(account)
            await MainActor.run {
                isSubmitting = false
                if success {
                    dismiss()
                } else {
                    errorMessage = "Failed to add account. Please check your inputs and try again."
                }
            }
        }
    }
}

#Preview {
    AddMT5AccountView(viewModel: MT5AccountViewModel())
}
