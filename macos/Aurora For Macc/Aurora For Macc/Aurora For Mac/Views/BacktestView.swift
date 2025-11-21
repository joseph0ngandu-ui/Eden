import SwiftUI

struct BacktestView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Backtest")
                            .font(.title)
                            .fontWeight(.bold)

                        Text("Test strategies on historical data")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .padding()

                VStack(spacing: 16) {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.system(size: 80))
                        .foregroundStyle(.orange)

                    Text("Backtesting Coming Soon")
                        .font(.title2)
                        .fontWeight(.semibold)

                    Text(
                        "Validate your strategies against historical market conditions before deploying them live."
                    )
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
                }
                .frame(maxWidth: 600)
                .padding(.top, 60)
            }
        }
    }
}

#Preview {
    BacktestView()
}
