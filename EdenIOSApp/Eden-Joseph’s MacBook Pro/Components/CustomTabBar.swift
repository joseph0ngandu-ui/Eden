//
//  CustomTabBar.swift
//  Eden
//
//  Custom tab bar component
//

import SwiftUI

struct CustomTabBar: View {
    @Binding var selectedTab: Int
    
    var body: some View {
        HStack(spacing: 0) {
            TabBarButton(icon: "chart.line.uptrend.xyaxis", text: "Overview", isSelected: selectedTab == 0) {
                selectedTab = 0
            }
            
            TabBarButton(icon: "chart.bar.fill", text: "Positions", isSelected: selectedTab == 1) {
                selectedTab = 1
            }
            
            TabBarButton(icon: "chart.pie.fill", text: "Analytics", isSelected: selectedTab == 2) {
                selectedTab = 2
            }
            
            TabBarButton(icon: "gearshape.fill", text: "Settings", isSelected: selectedTab == 3) {
                selectedTab = 3
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 24)
                .fill(Color.gray.opacity(0.15))
        )
    }
}

struct TabBarButton: View {
    let icon: String
    let text: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 20))
                
                Text(text)
                    .font(.system(size: 10))
            }
            .foregroundColor(isSelected ? .white : .gray)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(
                isSelected ?
                LinearGradient(
                    colors: [.purple, .blue],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .cornerRadius(16) : nil
            )
        }
    }
}

#Preview {
    CustomTabBar(selectedTab: .constant(0))
        .padding()
        .background(Color.black)
}
