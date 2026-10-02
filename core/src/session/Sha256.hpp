#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// Self-contained SHA-256 (FIPS 180-4), for the session manifest's model file
// hashes. The core has no crypto dependency and this is the only use.
class Sha256 {
public:
    Sha256();
    void        update(const void* data, size_t len);
    std::string hexDigest(); // finalises; call once

    // Hashes a file in blocks. Returns false if it cannot be opened or read.
    static bool hashFile(const std::string& path, std::string* hex_out, uint64_t* size_out);

private:
    void     transform(const uint8_t block[64]);
    uint32_t state_[8];
    uint8_t  buffer_[64];
    size_t   buffer_len_ = 0;
    uint64_t total_len_  = 0;
};
