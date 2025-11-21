import SwiftUI

struct MLTrainingView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                HStack {
                    VStack(alignment: .leading) {
                        Text("ML Training")
                            .font(.title)
                            .fontWeight(.bold)

                        Text("Train models to discover profitable trading strategies")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .padding()

                // Coming Soon Card
                VStack(spacing: 16) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 80))
                        .foregroundStyle(.blue)

                    Text("ML Training Coming Soon")
                        .font(.title2)
                        .fontWeight(.semibold)

                    Text(
                        "This feature will allow you to train Create ML models on historical data to automatically generate trading strategies."
                    )
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)

                    VStack(alignment: .leading, spacing: 12) {
                        Label("Load historical price data", systemImage: "chart.xyaxis.line")
                        Label(
                            "Select technical indicators as features",
                            systemImage: "slider.horizontal.3")
                        Label("Train regression or classification models", systemImage: "cpu")
                        Label(
                            "Generate strategy parameters from model", systemImage: "wand.and.stars"
                        )
                        Label("Backtest before deploying", systemImage: "checkmark.shield")
                    }
                    .padding()
                    .background(Color.blue.opacity(0.05))
                    .cornerRadius(12)
                }
                .frame(maxWidth: 600)
                .padding(.top, 60)
            }
        }
    }
}

#Preview {
    MLTrainingView()
}
