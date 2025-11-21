import Combine
import Foundation

@MainActor
class WebSocketService: NSObject, ObservableObject {
    static let shared = WebSocketService()

    @Published var isConnected: Bool = false
    @Published var positions: [Position] = []
    @Published var recentTrades: [Trade] = []
    @Published var connectionError: String?

    private var webSocket: URLSessionWebSocketTask?
    private var session: URLSession?
    private let apiService = APIService.shared

    private override init() {
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    func connect() {
        guard !isConnected else { return }

        // Parse base URL to get WebSocket endpoint
        guard let baseURL = URL(string: apiService.baseURL) else {
            connectionError = "Invalid base URL"
            return
        }

        // Convert https to wss
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        components?.scheme = baseURL.scheme == "https" ? "wss" : "ws"
        components?.path = "/ws/monitor"

        guard let wsURL = components?.url else {
            connectionError = "Failed to create WebSocket URL"
            return
        }

        var request = URLRequest(url: wsURL)
        request.timeoutInterval = 10

        webSocket = session?.webSocketTask(with: request)
        webSocket?.resume()

        receiveMessage()

        Task {
            await MainActor.run {
                self.isConnected = true
                self.connectionError = nil
            }
        }
    }

    func disconnect() {
        webSocket?.cancel(with: .goingAway, reason: nil)
        isConnected = false
    }

    private func receiveMessage() {
        webSocket?.receive { [weak self] result in
            guard let self = self else { return }

            switch result {
            case .success(let message):
                self.handleMessage(message)
                self.receiveMessage()  // Continue listening

            case .failure(let error):
                Task { @MainActor in
                    self.isConnected = false
                    self.connectionError = error.localizedDescription
                }

                // Attempt reconnection after 5 seconds
                Task {
                    try? await Task.sleep(nanoseconds: 5_000_000_000)
                    await MainActor.run {
                        self.connect()
                    }
                }
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            parseMessage(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                parseMessage(text)
            }
        @unknown default:
            break
        }
    }

    private func parseMessage(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }

        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601

            // Try to decode as WebSocket message envelope
            if let envelope = try? decoder.decode(WebSocketMessage.self, from: data) {
                Task { @MainActor in
                    switch envelope.type {
                    case "positions_update":
                        if let positions = envelope.data["positions"] as? [[String: Any]] {
                            self.updatePositions(from: positions)
                        }
                    case "trade_executed":
                        if let tradeData = envelope.data["trade"] as? [String: Any] {
                            self.addTrade(from: tradeData)
                        }
                    case "heartbeat":
                        // Just to keep connection alive
                        break
                    default:
                        print("Unknown message type: \(envelope.type)")
                    }
                }
            }
        } catch {
            print("Failed to parse WebSocket message: \(error)")
        }
    }

    private func updatePositions(from data: [[String: Any]]) {
        // Convert dictionary array to Position objects
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: data)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let newPositions = try decoder.decode([Position].self, from: jsonData)

            Task { @MainActor in
                self.positions = newPositions
            }
        } catch {
            print("Failed to decode positions: \(error)")
        }
    }

    private func addTrade(from data: [String: Any]) {
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: data)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let trade = try decoder.decode(Trade.self, from: jsonData)

            Task { @MainActor in
                self.recentTrades.insert(trade, at: 0)
                // Keep only last 50 trades
                if self.recentTrades.count > 50 {
                    self.recentTrades = Array(self.recentTrades.prefix(50))
                }
            }
        } catch {
            print("Failed to decode trade: \(error)")
        }
    }

    // WebSocket message envelope structure
    struct WebSocketMessage: Codable {
        let type: String
        let data: [String: Any]

        enum CodingKeys: String, CodingKey {
            case type
            case data
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            type = try container.decode(String.self, forKey: .type)

            // Decode data as generic dictionary
            if let dataDict = try? container.decode([String: AnyCodable].self, forKey: .data) {
                data = dataDict.mapValues { $0.value }
            } else {
                data = [:]
            }
        }
    }
}

// Helper to decode Any values
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodable($0) })
        default:
            try container.encodeNil()
        }
    }
}

// URLSessionWebSocketDelegate
extension WebSocketService: URLSessionWebSocketDelegate {
    nonisolated func urlSession(
        _ session: URLSession, webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        Task { @MainActor in
            self.isConnected = true
            self.connectionError = nil
        }
    }

    nonisolated func urlSession(
        _ session: URLSession, webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?
    ) {
        Task { @MainActor in
            self.isConnected = false
        }
    }
}
