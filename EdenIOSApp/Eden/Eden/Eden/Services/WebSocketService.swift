//
//  WebSocketService.swift
//  Eden
//
//  WebSocket service for real-time updates
//

import Foundation

class WebSocketService: NSObject, URLSessionWebSocketDelegate {
    static let shared = WebSocketService()
    
    private var webSocketTask: URLSessionWebSocketTask?
    private var wsURL: String {
        return APIEndpoints.wsBaseURL + "/ws/notifications"
    }
    
    var onMessage: ((String) -> Void)?
    var onConnected: (() -> Void)?
    var onDisconnected: (() -> Void)?
    
    private override init() {
        super.init()
    }
    
    // MARK: - Connection Management
    func connect() {
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: OperationQueue())
        guard let url = URL(string: wsURL) else {
            print("Invalid WebSocket URL")
            return
        }
        
        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()
        receiveMessage()
    }
    
    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
    }
    
    // MARK: - Message Handling
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    DispatchQueue.main.async {
                        self?.onMessage?(text)
                    }
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        DispatchQueue.main.async {
                            self?.onMessage?(text)
                        }
                    }
                @unknown default:
                    break
                }
                self?.receiveMessage()
                
            case .failure(let error):
                print("WebSocket receive error: \(error)")
            }
        }
    }
    
    func send(message: String) {
        let message = URLSessionWebSocketTask.Message.string(message)
        webSocketTask?.send(message) { error in
            if let error = error {
                print("WebSocket send error: \(error)")
            }
        }
    }
    
    // MARK: - URLSessionWebSocketDelegate
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        print("✓ WebSocket connected")
        DispatchQueue.main.async {
            self.onConnected?()
        }
    }
    
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        print("✗ WebSocket disconnected")
        DispatchQueue.main.async {
            self.onDisconnected?()
        }
    }
}
