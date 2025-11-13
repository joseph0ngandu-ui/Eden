//
//  MetricRow.swift
//  Eden
//
//  Analytics metric row component
//

import SwiftUI

struct MetricRow: View {
    let label: String
    let value: String
    var valueColor: Color = .white
    
    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 15))
                .foregroundColor(.gray)
            
            Spacer()
            
            Text(value)
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(valueColor)
        }
    }
}

#Preview {
    VStack {
        MetricRow(label: "Peak Balance", value: "$389.45")
        MetricRow(label: "Current Drawdown", value: "10.7%", valueColor: .red)
        MetricRow(label: "Profit Factor", value: "2.34", valueColor: .green)
    }
    .padding()
    .background(Color.black)
}
