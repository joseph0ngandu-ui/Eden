import SwiftUI

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    let trend: String?

    init(title: String, value: String, icon: String, color: Color = .blue, trend: String? = nil) {
        self.title = title
        self.value = value
        self.icon = icon
        self.color = color
        self.trend = trend
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(color)

                Spacer()

                if let trend = trend {
                    Text(trend)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(trend.hasPrefix("+") ? .green : .red)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)

                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(NSColor.controlBackgroundColor))
                .shadow(color: .black.opacity(0.05), radius: 4, x: 0, y: 2)
        )
    }
}

#Preview {
    HStack(spacing: 16) {
        StatCard(
            title: "Total P&L",
            value: "+$1,234.56",
            icon: "dollarsign.circle.fill",
            color: .green,
            trend: "+12.3%"
        )

        StatCard(
            title: "Active Positions",
            value: "5",
            icon: "chart.line.uptrend.xyaxis",
            color: .blue
        )

        StatCard(
            title: "Win Rate",
            value: "68.5%",
            icon: "target",
            color: .orange,
            trend: "+2.1%"
        )
    }
    .padding()
    .frame(width: 600)
}
