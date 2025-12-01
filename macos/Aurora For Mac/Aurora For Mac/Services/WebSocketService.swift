import Combine
import Foundation

class WebSocketService: ObservableObject {
    static let shared = WebSocketService()
    
    // Connection States
    @Published var isUpdatesConnected = false
    @Published var isTradesConnected = false
    
    // Data Publishers
    let botStatusSubject = PassthroughSubject<BotStatus, Never>()
    let tradeUpdateSubject = PassthroughSubject<[Trade], Never>()
    
    private var updatesSocket: URLSessionWebSocketTask?
    private var tradesSocket: URLSessionWebSocketTask?
    private let apiService = APIService.shared
    
    private var is intentionalDisconnect = false
    
    private init() {}
    
    // MARK: - Connection Management
    
    func connectAll() {
        guard let token = apiService.authToken else {
            print("⚠️ Cannot connect to WebSockets: No auth token")
            return
        }
        
        intentionalDisconnect = false
        connectUpdates(token: token)
        connectTrades(token: token)
    }
    
    func disconnectAll() {
        intentionalDisconnect = true
        updatesSocket?.cancel(with: .goingAway, reason: nil)
        tradesSocket?.cancel(with: .goingAway, reason: nil)
        updatesSocket = nil
        tradesSocket = nil
        isUpdatesConnected = false
        isTradesConnected = false
    }
    
    // MARK: - Updates Socket (Bot Status)
    
    private func connectUpdates(token: String) {
        let wsURL = getWebSocketURL(endpoint: "/ws/updates/\(token)")
        print("🔌 Connecting to Updates WS: \(wsURL)")
        
        let session = URLSession(configuration: .default)
        updatesSocket = session.webSocketTask(with: wsURL)
        updatesSocket?.resume()
        
        isUpdatesConnected = true
        listenForUpdates()
    }
    
    private func listenForUpdates() {
        updatesSocket?.receive { [weak self] result in
            guard let self = self, !self.is intentionalDisconnect else { return }
            
            switch result {
            case .success(let message):
                self.handleUpdatesMessage(message)
                self.listenForUpdates() // Keep listening
                
            case .failure(let error):
                print("❌ Updates WS Error: \(error)")
                DispatchQueue.main.async {
                    self.isUpdatesConnected = false
                }
                self.reconnectUpdates()
            }
        }
    }
    
    private func handleUpdatesMessage(_ message: URLSessionWebSocketTask.Message) {
        guard case .string(let text) = message,
              let data = text.data(using: .utf8) else { return }
        
        do {
            let response = try JSONDecoder().decode(WSResponse<BotStatus>.self, from: data)
            if response.type == "bot_status" {
                DispatchQueue.main.async {
                    self.botStatusSubject.send(response.data)
                }
            }
        } catch {
            print("⚠️ Failed to decode updates message: \(error)")
        }
    }
    
    private func reconnectUpdates() {
        guard !intentionalDisconnect else { return }
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) { [weak self] in
            if let token = self?.apiService.authToken {
                self?.connectUpdates(token: token)
            }
        }
    }
    
    // MARK: - Trades Socket
    
    private func connectTrades(token: String) {
        let wsURL = getWebSocketURL(endpoint: "/ws/trades/\(token)")
        print("🔌 Connecting to Trades WS: \(wsURL)")
        
        let session = URLSession(configuration: .default)
        tradesSocket = session.webSocketTask(with: wsURL)
        tradesSocket?.resume()
        
        isTradesConnected = true
        listenForTrades()
    }
    
    private func listenForTrades() {
        tradesSocket?.receive { [weak self] result in
            guard let self = self, !self.is intentionalDisconnect else { return }
            
            switch result {
            case .success(let message):
                self.handleTradesMessage(message)
                self.listenForTrades()
                
            case .failure(let error):
                print("❌ Trades WS Error: \(error)")
                DispatchQueue.main.async {
                    self.isTradesConnected = false
                }
                self.reconnectTrades()
            }
        }
    }
    
    private func handleTradesMessage(_ message: URLSessionWebSocketTask.Message) {
        guard case .string(let text) = message,
              let data = text.data(using: .utf8) else { return }
        
        do {
            // Try decoding as list of trades
            let response = try JSONDecoder().decode(WSResponse<[Trade]>.self, from: data)
            if response.type == "trade_update" {
                DispatchQueue.main.async {
                    self.tradeUpdateSubject.send(response.data)
                }
            }
        } catch {
            print("⚠️ Failed to decode trades message: \(error)")
        }
    }
    
    private func reconnectTrades() {
        guard !intentionalDisconnect else { return }
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) { [weak self] in
            if let token = self?.apiService.authToken {
                self?.connectTrades(token: token)
            }
        }
    }
    
    // MARK: - Helpers
    
    private func getWebSocketURL(endpoint: String) -> URL {
        let baseURL = apiService.baseURL
        let wsScheme = baseURL.starts(with: "https") ? "wss" : "ws"
        
        // Strip scheme from baseURL
        let hostPath = baseURL.replacingOccurrences(of: "https://", with: "")
                              .replacingOccurrences(of: "http://", with: "")
        
        return URL(string: "\(wsScheme)://\(hostPath)\(endpoint)")!
    }
}

// MARK: - WS Response Wrapper

struct WSResponse<T: Decodable>: Decodable {
    let type: String
    let data: T
}
