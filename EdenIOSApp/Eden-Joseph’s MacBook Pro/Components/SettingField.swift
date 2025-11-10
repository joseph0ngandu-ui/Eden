//
//  SettingField.swift
//  Eden
//
//  Settings input field component
//

import SwiftUI

struct SettingField: View {
    let label: String
    @Binding var text: String
    var isSecure: Bool = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.system(size: 13))
                .foregroundColor(.gray)
            
            if isSecure {
                SecureField("Enter \(label.lowercased())", text: $text)
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(12)
                    .foregroundColor(.white)
            } else {
                TextField("Enter \(label.lowercased())", text: $text)
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(12)
                    .foregroundColor(.white)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
            }
        }
    }
}

#Preview {
    VStack {
        SettingField(label: "Webhook URL", text: .constant("https://example.com"))
        SettingField(label: "API Key", text: .constant("secret"), isSecure: true)
    }
    .padding()
    .background(Color.black)
}
