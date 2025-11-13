//
//  StrategiesView.swift
//  Eden
//
//  Displays discovered strategies with ability to activate/deactivate.
//

import SwiftUI

struct StrategiesView: View {
    @State private var strategies: [StrategyItem] = []
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            Group {
                if isLoading {
                    ProgressView("Loading strategies...")
                } else if let errorMessage = errorMessage {
                    Text(errorMessage).foregroundColor(.red)
                } else if strategies.isEmpty {
                    VStack(spacing: 12) {
                        Text("No strategies yet")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Button("Discover New Strategy") {
                            discover()
                        }
                    }
                } else {
                    List {
                        ForEach(strategies) { s in
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(s.name ?? s.id).font(.headline)
                                    HStack(spacing: 8) {
                                        if let t = s.type { Text(t).font(.caption).foregroundColor(.secondary) }
                                        if s.validated == true { Text("VALIDATED").font(.caption2).padding(4).background(Color.green.opacity(0.2)).cornerRadius(4) }
                                    }
                                }
                                Spacer()
                                Toggle("", isOn: Binding(
                                    get: { s.is_active ?? false },
                                    set: { val in toggle(s, val) }
                                ))
                                .labelsHidden()
                            }
                        }
                    }
                }
            }
            .navigationTitle("Strategies")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: reload) { Image(systemName: "arrow.clockwise") }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: discover) { Image(systemName: "sparkles") }
                }
            }
            .onAppear(perform: reload)
        }
    }

    private func reload() {
        isLoading = true
        StrategiesService.shared.list { result in
            DispatchQueue.main.async {
                isLoading = false
                switch result {
                case .success(let items): self.strategies = items
                case .failure(let err): self.errorMessage = err.localizedDescription
                }
            }
        }
    }

    private func toggle(_ s: StrategyItem, _ on: Bool) {
        if on {
            StrategiesService.shared.activate(id: s.id) { _ in reload() }
        } else {
            StrategiesService.shared.deactivate(id: s.id) { _ in reload() }
        }
    }

    private func discover() {
        guard let url = URL(string: APIEndpoints.Strategies.discover) else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        NetworkManager.shared.dataTask(with: req) { _, _, _ in reload() }.resume()
    }
}

struct StrategiesView_Previews: PreviewProvider {
    static var previews: some View {
        StrategiesView()
    }
}
