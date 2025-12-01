import Foundation
import UserNotifications
import UIKit

class NotificationService: NSObject, ObservableObject {
    static let shared = NotificationService()
    private let apiService = APIService.shared
    
    @Published var isRegistered = false
    @Published var errorMessage: String?
    
    override private init() {
        super.init()
    }
    
    func requestPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if granted {
                DispatchQueue.main.async {
                    NSApplication.shared.registerForRemoteNotifications()
                }
            } else if let error = error {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to request permissions: \(error.localizedDescription)"
                }
            }
        }
    }
    
    func registerDevice(token: Data) async {
        let tokenString = token.map { String(format: "%02.2hhx", $0) }.joined()
        
        struct DeviceRegistration: Codable {
            let token: String
            let platform: String = "macos"
        }
        
        do {
            let registration = DeviceRegistration(token: tokenString)
            let encoder = JSONEncoder()
            let data = try encoder.encode(registration)
            
            _ = try await apiService.performRequest(
                endpoint: "/device/register",
                method: "POST",
                body: data
            )
            
            await MainActor.run {
                self.isRegistered = true
            }
        } catch {
            await MainActor.run {
                self.errorMessage = "Failed to register device: \(error.localizedDescription)"
            }
        }
    }
    
    func unregisterDevice(token: String) async {
        do {
            _ = try await apiService.performRequest(
                endpoint: "/device/\(token)",
                method: "DELETE"
            )
            
            await MainActor.run {
                self.isRegistered = false
            }
        } catch {
            await MainActor.run {
                self.errorMessage = "Failed to unregister device: \(error.localizedDescription)"
            }
        }
    }
}

extension NotificationService: UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }
}
