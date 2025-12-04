import Foundation

struct StrategyConfig: Codable, Equatable {
    var enabledSymbols: [String]

    enum CodingKeys: String, CodingKey {
        case enabledSymbols = "enabled_symbols"
    }
}
