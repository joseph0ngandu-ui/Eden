import SwiftUI
import UserNotifications

class NotificationManager: ObservableObject {
    static let shared = NotificationManager()

    @Published var hasPermission: Bool = false

    private init() {
        checkPermissions()
    }

        requestPermission()
    }
    
    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            DispatchQueue.main.async {
                self.permissionGranted = granted
            }
        }
    }
    
    func sendTradeNotification(trade: Trade) {
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
