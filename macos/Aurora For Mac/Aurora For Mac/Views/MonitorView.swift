import SwiftUI

struct MonitorView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Monitor")
                            .font(.title)
                            .fontWeight(.bold)

                        Text("Track live strategy performance")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .padding()

                VStack(spacing: 16) {
                    Image(systemName: "chart.xyaxis.line")
                        .font(.system(size: 80))
                        .foregroundStyle(.green)

                    Text("Monitoring Coming Soon")
                        .font(.title2)
                        .fontWeight(.semibold)

                    Text("Real-time monitoring of your active strategies and bot performance.")
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
    MonitorView()
}
