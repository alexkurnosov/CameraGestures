#include "TaskBundleParser.hpp"
#include <fstream>
#include <cstring>

// ZIP structure constants
static constexpr uint32_t kLocalFileHeaderSig    = 0x04034b50;
static constexpr uint32_t kCentralDirHeaderSig   = 0x02014b50;
static constexpr uint32_t kEndOfCentralDirSig    = 0x06054b50;

static uint16_t read16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static uint32_t read32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

TaskBundleParser::TaskBundleParser(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return;
    auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    data_.resize(sz);
    f.read(reinterpret_cast<char*>(data_.data()), static_cast<std::streamsize>(sz));
}

std::vector<uint8_t> TaskBundleParser::extract(const std::string& name) const {
    if (data_.empty()) return {};
    const uint8_t* d   = data_.data();
    const size_t   sz  = data_.size();

    // Locate End-Of-Central-Directory record (search backwards from end)
    size_t eocd_off = 0;
    bool   found    = false;
    for (size_t i = sz >= 22 ? sz - 22 : 0; ; --i) {
        if (i + 4 <= sz && read32(d + i) == kEndOfCentralDirSig) {
            eocd_off = i;
            found    = true;
            break;
        }
        if (i == 0) break;
    }
    if (!found) return {};

    uint16_t num_entries  = read16(d + eocd_off + 10);
    uint32_t cd_offset    = read32(d + eocd_off + 16);

    // Walk central directory entries
    size_t cd_pos = cd_offset;
    for (uint16_t i = 0; i < num_entries; ++i) {
        if (cd_pos + 46 > sz) break;
        if (read32(d + cd_pos) != kCentralDirHeaderSig) break;

        uint16_t fname_len  = read16(d + cd_pos + 28);
        uint16_t extra_len  = read16(d + cd_pos + 30);
        uint16_t comment_len= read16(d + cd_pos + 32);
        uint32_t local_off  = read32(d + cd_pos + 42);

        std::string entry_name(reinterpret_cast<const char*>(d + cd_pos + 46), fname_len);
        cd_pos += 46 + fname_len + extra_len + comment_len;

        if (entry_name != name) continue;

        // Found — jump to local file header
        size_t lf = local_off;
        if (lf + 30 > sz) return {};
        if (read32(d + lf) != kLocalFileHeaderSig) return {};

        uint16_t compress   = read16(d + lf + 8);
        uint32_t comp_size  = read32(d + lf + 18);
        uint32_t uncomp_size= read32(d + lf + 22);
        uint16_t lf_name    = read16(d + lf + 26);
        uint16_t lf_extra   = read16(d + lf + 28);

        size_t data_start = lf + 30 + lf_name + lf_extra;
        if (data_start + comp_size > sz) return {};

        if (compress != 0) {
            // Should never happen for hand_landmarker.task (all STORED)
            return {};
        }

        return std::vector<uint8_t>(d + data_start, d + data_start + uncomp_size);
    }
    return {};
}
