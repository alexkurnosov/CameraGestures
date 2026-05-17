import Foundation
import Security

/// Stores, loads, and deletes the server JWT in the iOS Keychain.
///
/// Uses kSecClassGenericPassword under service "GestureModelAPI".
/// No Keychain Sharing entitlement is required for single-app use.
final class TokenStorage {

    private let service = "GestureModelAPI"
    private let account = "jwt_token"

    func save(_ token: String) {
        let data = Data(token.utf8)
        // Delete any existing item first so SecItemAdd always succeeds.
        SecItemDelete(query() as CFDictionary)
        var attributes = query()
        attributes[kSecValueData] = data
        SecItemAdd(attributes as CFDictionary, nil)
    }

    func load() -> String? {
        var q = query()
        q[kSecReturnData]  = true
        q[kSecMatchLimit]  = kSecMatchLimitOne
        var result: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    func delete() {
        SecItemDelete(query() as CFDictionary)
    }

    // MARK: - Private

    private func query() -> [CFString: Any] {
        [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: account,
        ]
    }
}
