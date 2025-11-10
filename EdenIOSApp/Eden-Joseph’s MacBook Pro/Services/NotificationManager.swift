//
//  NotificationManager.swift
//  Eden
//
//  Push notification manager
//

import UserNotifications
import UIKit

class NotificationManager: NSObject {
    static let shared = NotificationManager()
    
    private override init() {
        super.init()
    }
    
    // MARK: - Request Authorization
    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("Notification permission error: \(error)")
                completion(false)
                return
            }
            
            if granted {
                print("✓ Notification permission granted")
                DispatchQueue.main.async {
                    UIApplication.shared.registerForRemoteNotifications()
                }
            }
            
            completion(granted)
        }
    }
    
    // MARK: - Schedule Local Notification
    func scheduleTradeNotification(symbol: String, pnl: Double, type: String) {
        let content = UNMutableNotificationContent()
        content.title = "Eden Trade Alert"
        content.body = "\(type) on \(symbol): \(pnl >= 0 ? "+" : "")\(String(format: "%.2f", pnl))"
        content.sound = .default
        content.badge = 1
        
        // Add custom data
        content.userInfo = [
            "symbol": symbol,
            "pnl": pnl,
            "type": type
        ]
        
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: trigger
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Error scheduling notification: \(error)")
            }
        }
    }
    
    // MARK: - Schedule Bot Status Notification
    func scheduleBotStatusNotification(isRunning: Bool, message: String) {
        let content = UNMutableNotificationContent()
        content.title = isRunning ? "Eden Bot Started" : "Eden Bot Stopped"
        content.body = message
        content.sound = .default
        
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(
            identifier: "bot-status-\(UUID().uuidString)",
            content: content,
            trigger: trigger
        )
        
        UNUserNotificationCenter.current().add(request)
    }
    
    // MARK: - Clear Badge
    func clearBadge() {
        UIApplication.shared.applicationIconBadgeNumber = 0
    }
    
    // MARK: - Handle Device Token
    func handleDeviceToken(_ deviceToken: Data) {
        let tokenParts = deviceToken.map { data in String(format: "%02.2hhx", data) }
        let token = tokenParts.joined()
        print("Device Token: \(token)")
        
        // TODO: Send token to your server
        // APIService.shared.registerDeviceToken(token)
    }
}

// MARK: - UNUserNotificationCenterDelegate
extension NotificationManager: UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        // Show notification even when app is in foreground
        completionHandler([.banner, .sound, .badge])
    }
    
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let userInfo = response.notification.request.content.userInfo
        
        // Handle notification tap
        if let symbol = userInfo["symbol"] as? String {
            print("User tapped notification for \(symbol)")
            // TODO: Navigate to specific position/trade
        }
        
        completionHandler()
    }
}
