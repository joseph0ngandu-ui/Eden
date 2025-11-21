import Combine
import Foundation
import SwiftUI
import UserNotifications

class NotificationManager: ObservableObject {
    static let shared = NotificationManager()

    @Published var permissionGranted = false

    private init() {
        requestPermission()
    }

    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) {
            granted, error in
            DispatchQueue.main.async {
                self.permissionGranted = granted
            }
        }
    }

    func sendTradeNotification(trade: Trade) {
        guard permissionGranted else { return }

        let content = UNMutableNotificationContent()
        content.title = "New Trade Executed"
        content.body = "\(trade.type.rawValue) \(trade.symbol) @ \(trade.price)"
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    func sendPositionUpdate(position: Position) {
        guard permissionGranted else { return }

        let content = UNMutableNotificationContent()
        content.title = "Position Update"
        content.body = "\(position.symbol): \(position.formattedPnl)"
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    func sendStatusNotification(isConnected: Bool) {
        guard permissionGranted else { return }

        let content = UNMutableNotificationContent()
        content.title = "Connection Status"
        content.body = isConnected ? "Connected to Server" : "Disconnected from Server"
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }
}
