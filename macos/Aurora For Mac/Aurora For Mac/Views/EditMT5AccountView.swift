import SwiftUI

struct EditMT5AccountView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var viewModel: MT5AccountViewModel
    let account: MT5Account
    
    @State private var password = ""
    @State private var server: String
    @State private var broker: String
    @State private var isActive: Bool
    @State private var isSubmitting = false
    @State private var errorMessage: String?
    
    init(account: MT5Account, viewModel: MT5AccountViewModel) {
        self.account = account
        self.viewModel = viewModel
        _server = State(initialValue: account.server)
        _broker = State(initialValue: account.broker)
        _isActive = State(initialValue: account.isActive)
    }
    
    var body: some View {
        NavigationStack {
            Form {
                Section("Account Details") {
                    LabeledContent("Account Number", value: account.accountNumber)
                        .foregroundColor(.secondary)
                    
                    SecureField("New Password (Optional)", text: $password)
                        .textContentType(.password)
                }
                
                Section("Broker Details") {
                    TextField("Server", text: $server)
                        .textContentType(.URL)
                        .autocorrectionDisabled()
                    
                    TextField("Broker Name", text: $broker)
                }
                
                Section("Status") {
                    Toggle("Active", isOn: $isActive)
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
            .navigationTitle("Edit Account")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        submit()
                    }
                    .disabled(server.isEmpty || broker.isEmpty || isSubmitting)
                }
            }
            .overlay {
                if isSubmitting {
                    ProgressView()
                }
            }
        }
        .frame(width: 400, height: 450)
    }
    
    private func submit() {
        isSubmitting = true
        errorMessage = nil
        
        let update = MT5AccountUpdate(
            password: password.isEmpty ? nil : password,
            server: server,
            broker: broker,
            isActive: isActive
        )
        
        Task {
            let success = await viewModel.updateAccount(id: account.id, data: update)
            await MainActor.run {
                isSubmitting = false
                if success {
                    dismiss()
                } else {
                    errorMessage = "Failed to update account."
                }
            }
        }
    }
}

#Preview {
    EditMT5AccountView(
        account: MT5Account(
            id: 1,
            accountNumber: "123456",
            server: "MetaQuotes-Demo",
            broker: "MetaQuotes",
            isPrimary: true,
            isActive: true,
            balance: 10000.0,
            equity: 10000.0,
            createdAt: Date(),
            updatedAt: Date()
        ),
        viewModel: MT5AccountViewModel()
    )
}
