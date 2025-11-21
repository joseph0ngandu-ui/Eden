import Charts
import SwiftUI

struct MLTrainingView: View {
    @EnvironmentObject var apiService: APIService
    @State private var selectedSymbol = "Volatility 75 Index"
    @State private var selectedModel = "LSTM"
    @State private var selectedDataLength = 1000
    @State private var epochs = 50
    @State private var learningRate = 0.001
    @State private var isTraining = false
    @State private var trainingStatus: String = "Idle"
    @State private var progress: Double = 0.0
    @State private var accuracy: Double = 0.0
    @State private var lossHistory: [Double] = []
    @State private var estimatedTime: String = "2 min"

    let symbols = ["Volatility 75 Index", "Volatility 100 Index", "XAUUSD", "EURUSD"]
    let models = ["LSTM", "Transformer"]
    let dataLengths = [1000, 5000, 10000]

    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Text("ML Training Center")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                Spacer()
                if isTraining {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            .padding()

            ScrollView {
                VStack(spacing: 24) {
                    // Configuration Card
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Configuration")
                            .font(.headline)
                            .foregroundColor(.secondary)

                        HStack(spacing: 20) {
                            VStack(alignment: .leading) {
                                Text("Symbol")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("Symbol", selection: $selectedSymbol) {
                                    ForEach(symbols, id: \.self) { symbol in
                                        Text(symbol).tag(symbol)
                                    }
                                }
                                .pickerStyle(.menu)
                            }

                            VStack(alignment: .leading) {
                                Text("Model")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("Model", selection: $selectedModel) {
                                    ForEach(models, id: \.self) { model in
                                        Text(model).tag(model)
                                    }
                                }
                                .pickerStyle(.segmented)
                            }
                        }

                        HStack(spacing: 20) {
                            VStack(alignment: .leading) {
                                Text("Data Length (Candles)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("Data Length", selection: $selectedDataLength) {
                                    ForEach(dataLengths, id: \.self) { length in
                                        Text("\(length)").tag(length)
                                    }
                                }
                                .pickerStyle(.segmented)
                            }

                            VStack(alignment: .leading) {
                                Text("Estimated Time")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(calculateEstimatedTime())
                                    .font(.system(.body, design: .monospaced))
                                    .foregroundColor(.accentColor)
                            }
                        }

                        Divider()

                        HStack {
                            VStack(alignment: .leading) {
                                Text("Epochs: \(epochs)")
                                Slider(
                                    value: Binding(
                                        get: { Double(epochs) }, set: { epochs = Int($0) }),
                                    in: 10...200, step: 10)
                            }

                            VStack(alignment: .leading) {
                                Text("Learning Rate: \(learningRate, specifier: "%.4f")")
                                Slider(value: $learningRate, in: 0.0001...0.01, step: 0.0001)
                            }
                        }
                    }
                    .padding()
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.white.opacity(0.1), lineWidth: 1)
                    )

                    // Action Button
                    Button(action: startTraining) {
                        HStack {
                            Image(systemName: isTraining ? "stop.fill" : "play.fill")
                            Text(isTraining ? "Stop Training" : "Start Training")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isTraining ? Color.red.opacity(0.2) : Color.accentColor)
                        .foregroundColor(isTraining ? .red : .white)
                        .cornerRadius(10)
                    }
                    .buttonStyle(.plain)
                    .disabled(isTraining)  // Disable for now to prevent multiple requests

                    // Status & Metrics
                    if isTraining || !lossHistory.isEmpty {
                        VStack(alignment: .leading, spacing: 16) {
                            Text("Training Progress")
                                .font(.headline)
                                .foregroundColor(.secondary)

                            HStack {
                                StatBox(title: "Status", value: trainingStatus)
                                StatBox(
                                    title: "Accuracy", value: String(format: "%.2f%%", accuracy))
                                StatBox(
                                    title: "Loss",
                                    value: String(format: "%.4f", lossHistory.last ?? 0.0))
                            }

                            if !lossHistory.isEmpty {
                                Chart(Array(lossHistory.enumerated()), id: \.offset) {
                                    index, loss in
                                    LineMark(
                                        x: .value("Epoch", index),
                                        y: .value("Loss", loss)
                                    )
                                    .interpolationMethod(.catmullRom)
                                    .foregroundStyle(Color.accentColor.gradient)

                                    AreaMark(
                                        x: .value("Epoch", index),
                                        y: .value("Loss", loss)
                                    )
                                    .interpolationMethod(.catmullRom)
                                    .foregroundStyle(
                                        LinearGradient(
                                            colors: [
                                                Color.accentColor.opacity(0.3),
                                                Color.accentColor.opacity(0.0),
                                            ],
                                            startPoint: .top,
                                            endPoint: .bottom
                                        )
                                    )
                                }
                                .frame(height: 200)
                                .chartYAxis {
                                    AxisMarks(position: .leading)
                                }
                            }
                        }
                        .padding()
                        .background(Color(nsColor: .controlBackgroundColor))
                        .cornerRadius(12)
                    }
                }
                .padding()
            }
        }
        .onReceive(Timer.publish(every: 2, on: .main, in: .common).autoconnect()) { _ in
            if isTraining {
                checkStatus()
            }
        }
    }

    private func calculateEstimatedTime() -> String {
        // Rough estimation: 0.1s per candle per epoch (very rough)
        let baseTime = Double(selectedDataLength) * Double(epochs) * 0.0005
        let modelFactor = selectedModel == "Transformer" ? 1.5 : 1.0
        let totalSeconds = baseTime * modelFactor

        if totalSeconds < 60 {
            return String(format: "%.0f sec", totalSeconds)
        } else {
            return String(format: "%.1f min", totalSeconds / 60)
        }
    }

    private func startTraining() {
        isTraining = true
        trainingStatus = "Initializing..."
        lossHistory = []
        accuracy = 0.0

        Task {
            do {
                try await apiService.startMLTraining(
                    symbol: selectedSymbol,
                    modelType: selectedModel,
                    dataLength: selectedDataLength,
                    epochs: epochs,
                    learningRate: learningRate
                )
                trainingStatus = "Training Started"
            } catch {
                trainingStatus = "Failed: \(error.localizedDescription)"
                isTraining = false
            }
        }
    }

    private func checkStatus() {
        Task {
            do {
                let status = try await apiService.getMLStatus(symbol: selectedSymbol)
                if let state = status["status"] as? String {
                    if state == "completed" {
                        isTraining = false
                        trainingStatus = "Completed"
                        if let metrics = status["metrics"] as? [String: Any] {
                            accuracy = metrics["accuracy"] as? Double ?? 0.0
                            if let finalLoss = metrics["final_loss"] as? Double {
                                lossHistory.append(finalLoss)
                            }
                        }
                    } else if state == "failed" {
                        isTraining = false
                        trainingStatus = "Failed"
                    } else {
                        trainingStatus = "Training..."
                        // In a real app, we'd stream loss history here
                    }
                }
            } catch {
                print("Status check failed: \(error)")
            }
        }
    }
}

struct StatBox: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.headline)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.white.opacity(0.05))
        .cornerRadius(8)
    }
}
