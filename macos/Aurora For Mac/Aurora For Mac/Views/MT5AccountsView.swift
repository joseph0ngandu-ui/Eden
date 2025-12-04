import SwiftUI

struct MT5AccountsView: View {
    @StateObject private var viewModel = MT5AccountViewModel()
    @State private var showingAddSheet = false
    @State private var editingAccount: MT5Account?
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("MT5 Accounts")
                    .font(.title)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button {
                    showingAddSheet = true
                } label: {
                    Label("Add Account", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
                
                Button {
                    Task {
                        await viewModel.loadAccounts()
                    }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
            .padding()
            
            Divider()
            
            // Content
            if viewModel.isLoading && viewModel.accounts.isEmpty {
                ProgressView("Loading accounts...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.accounts.isEmpty {
                ContentUnavailableView {
                    Label("No Accounts", systemImage: "server.rack")
                } description: {
                    Text("Connect your MetaTrader 5 account to start trading")
                } actions: {
                    Button("Add Account") {
                        showingAddSheet = true
                    }
                }
            } else {
                List {
                    ForEach(viewModel.accounts) { account in
                        MT5AccountRow(account: account, viewModel: viewModel) {
                            editingAccount = account
                        }
                    }
                }
                .listStyle(.inset)
            }
            
            if let error = viewModel.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                    Text(error)
                    Spacer()
                    Button("Dismiss") {
                        viewModel.errorMessage = nil
                    }
                }
                .padding()
                .background(Color.red.opacity(0.1))
            }
        }
        .sheet(isPresented: $showingAddSheet) {
            AddMT5AccountView(viewModel: viewModel)
        }
        .sheet(item: $editingAccount) { account in
            EditMT5AccountView(account: account, viewModel: viewModel)
        }
        .task {
            await viewModel.loadAccounts()
        }
    }
}

struct MT5AccountRow: View {
    let account: MT5Account
    let viewModel: MT5AccountViewModel
    let onEdit: () -> Void
    
    @State private var showDeleteConfirmation = false
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(account.accountNumber)
                        .font(.headline)
                    
                    if account.isPrimary {
                        Text("PRIMARY")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.blue.opacity(0.2))
                            .foregroundColor(.blue)
                            .cornerRadius(4)
                    }
                    
                    if !account.isActive {
                        Text("INACTIVE")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.gray.opacity(0.2))
                            .foregroundColor(.gray)
                            .cornerRadius(4)
                    }
                }
                
                HStack(spacing: 12) {
                    Label(account.broker, systemImage: "building.2")
                    Label(account.server, systemImage: "server.rack")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(account.balance, format: .currency(code: "USD"))
                    .fontWeight(.medium)
                
                Text("Equity: \(account.equity.formatted(.currency(code: "USD")))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.trailing, 8)
            
            // Actions
            Menu {
                Button {
                    onEdit()
                } label: {
                    Label("Edit Details", systemImage: "pencil")
                }
                
                if !account.isPrimary {
                    Button {
                        Task {
                            await viewModel.setPrimary(account)
                        }
                    } label: {
                        Label("Set as Primary", systemImage: "star")
                    }
                }
                
                Divider()
                
                Button(role: .destructive) {
                    showDeleteConfirmation = true
                } label: {
                    Label("Delete Account", systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(.title2)
                    .foregroundColor(.secondary)
            }
            .menuStyle(.borderlessButton)
            .frame(width: 30)
        }
        .padding(.vertical, 8)
        .alert("Delete Account?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                Task {
                    await viewModel.deleteAccount(account)
                }
            }
        } message: {
            Text("Are you sure you want to delete account \(account.accountNumber)? This action cannot be undone.")
        }
    }
}

#Preview {
    MT5AccountsView()
}
