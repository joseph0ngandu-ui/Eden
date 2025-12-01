import Foundation

struct MT5Account: Codable, Identifiable, Hashable {
    let id: Int
    let accountNumber: String
    var server: String
    var broker: String
    var isPrimary: Bool
    var isActive: Bool
    var balance: Double
    var equity: Double
    var createdAt: Date
    var updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case id
        case accountNumber = "account_number"
        case server
        case broker
        case isPrimary = "is_primary"
        case isActive = "is_active"
        case balance
        case equity
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

struct MT5AccountCreate: Codable {
    let accountNumber: String
    let password: String
    let server: String
    let broker: String
    let isPrimary: Bool
    let isActive: Bool

    enum CodingKeys: String, CodingKey {
        case accountNumber = "account_number"
        case password
        case server
        case broker
        case isPrimary = "is_primary"
        case isActive = "is_active"
    }
}

struct MT5AccountUpdate: Codable {
    let password: String?
    let server: String
    let broker: String
    let isActive: Bool

    enum CodingKeys: String, CodingKey {
        case password
        case server
        case broker
        case isActive = "is_active"
    }
}
