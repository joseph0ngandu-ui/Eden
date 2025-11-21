import SwiftUI
import UserNotifications

class NotificationManager: ObservableObject {
    static let shared = NotificationManager()

    @Published var hasPermission: Bool = false

    private init() {
        checkPermissions()
    }

    func requestPermissions() {
        let center = UNUserNotificationCenter.current()

        center.requestAuthorization(options: [.alert, .sound, .badge]) {
            [weak self] granted, error in
            DispatchQueue.main.async {
                self?.hasPermission = granted
            }

            if let error = error {
                print("Notification permission error: \(error)")
            }
        }
    }

    func checkPermissions() {
        let center = UNUserNotificationCenter.current()

        center.getNotificationSettings { [weak self] settings in
            DispatchQueue.main.async {
                self?.hasPermission = settings.authorizationStatus == .authorized
            }
        }
    }

    func sendTradeNotification(trade: Trade) {
        guard hasPermission else { return }

        let content = UNMutableNotificationContent()
        content.title = "Trade Executed"
        content.subtitle = "\(trade.side.rawValue) \(trade.symbol)"
        content.body =
            "Quantity: \(String(format: "%.4f", trade.quantity)) @ $\(String(format: "%.2f", trade.price))"
        content.sound = .default

        // Add badge if profitable
        if let pnl = trade.realizedPnl, pnl > 0 {
            content.badge = NSNumber(value: 1)
        }

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil  // Deliver immediately
        )

        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to send notification: \(error)")
            }
        }
    }

    func sendPositionAlert(position: Position, message: String) {
        guard hasPermission else { return }

        let content = UNMutableNotificationContent()
        content.title = "Position Alert"
        content.subtitle = position.symbol
        content.body = message
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )

        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to send notification: \(error)")
            }
        }
    }

    func sendConnectionAlert(isConnected: Bool) {
        guard hasPermission else { return }

        let content = UNMutableNotificationContent()
        content.title = isConnected ? "Connected" : "Connection Lost"
        content.body =
            isConnected
            ? "Successfully connected to trading bot"
            : "Lost connection to trading bot. Trying to reconnect..."
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )

        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Failed to send notification: \(error)")
            }
        }
    }
}
