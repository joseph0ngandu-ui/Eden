import SwiftUI

struct StrategyEditorView: View {
    @Environment(\.dismiss) var dismiss
    @State private var strategy: Strategy
    let onSave: (Strategy) -> Void

    init(strategy: Strategy, onSave: @escaping (Strategy) -> Void) {
        _strategy = State(initialValue: strategy)
        self.onSave = onSave
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Basic Info") {
                    TextField("Name", text: $strategy.name)
                    TextField(
                        "Description",
                        text: Binding(
                            get: { strategy.description ?? "" },
                            set: { strategy.description = $0.isEmpty ? nil : $0 }
                        ), axis: .vertical
                    )
                    .lineLimit(3...6)
                }

                Section("Parameters") {
                    HStack {
                        Text("Risk per Trade")
                        Spacer()
                        TextField("", value: $strategy.parametersModel.riskPerTrade, format: .number)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 100)
                        Text("%")
                    }

                    Stepper(
                        "Max Positions: \(strategy.parametersModel.maxPositions)",
                        value: $strategy.parametersModel.maxPositions, in: 1...10)

                    HStack {
                        Text("Stop Loss")
                        Spacer()
                        TextField("", value: $strategy.parametersModel.stopLossPercent, format: .number)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 100)
                        Text("%")
                    }

                    HStack {
                        Text("Take Profit")
                        Spacer()
                        TextField(
                            "", value: $strategy.parametersModel.takeProfitPercent, format: .number
                        )
                        .multilineTextAlignment(.trailing)
                        .frame(width: 100)
                        Text("%")
                    }

                    Picker("Timeframe", selection: $strategy.parametersModel.timeframe) {
                        Text("5 Minutes").tag("M5")
                        Text("15 Minutes").tag("M15")
                        Text("1 Hour").tag("H1")
                        Text("4 Hours").tag("H4")
                        Text("1 Day").tag("D1")
                    }
                }

                Section("Indicators") {
                    ForEach(strategy.indicators.indices, id: \.self) { index in
                        TextField("Indicator", text: $strategy.indicators[index])
                    }

                    Button("Add Indicator") {
                        strategy.indicators.append("")
                    }
                }

                Section("Entry Conditions - Long") {
                    ForEach(strategy.conditions.entryLong.indices, id: \.self) { index in
                        TextField("Condition", text: $strategy.conditions.entryLong[index])
                    }

                    Button("Add Condition") {
                        strategy.conditions.entryLong.append("")
                    }
                }

                Section("Entry Conditions - Short") {
                    ForEach(strategy.conditions.entryShort.indices, id: \.self) { index in
                        TextField("Condition", text: $strategy.conditions.entryShort[index])
                    }

                    Button("Add Condition") {
                        strategy.conditions.entryShort.append("")
                    }
                }

                Section("Exit Conditions - Long") {
                    ForEach(strategy.conditions.exitLong.indices, id: \.self) { index in
                        TextField("Condition", text: $strategy.conditions.exitLong[index])
                    }

                    Button("Add Condition") {
                        strategy.conditions.exitLong.append("")
                    }
                }

                Section("Exit Conditions - Short") {
                    ForEach(strategy.conditions.exitShort.indices, id: \.self) { index in
                        TextField("Condition", text: $strategy.conditions.exitShort[index])
                    }

                    Button("Add Condition") {
                        strategy.conditions.exitShort.append("")
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Edit Strategy")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Save & Upload") {
                        onSave(strategy)
                    }
                    .disabled(strategy.name.isEmpty)
                }
            }
        }
        .frame(minWidth: 600, minHeight: 700)
    }
}

#Preview {
    StrategyEditorView(strategy: Strategy(name: "Test Strategy")) { _ in }
}
