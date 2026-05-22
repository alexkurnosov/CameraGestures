#pragma once
#include <string>
#include <vector>
#include <cstdint>

// Minimal ZIP reader for the hand_landmarker.task bundle.
// Only handles STORED (compress_type 0) files; the .task format uses no compression.
class TaskBundleParser {
public:
    explicit TaskBundleParser(const std::string& path);

    // Returns true if the archive was opened successfully.
    bool isOpen() const { return !data_.empty(); }

    // Extract file by name. Returns empty vector if not found.
    std::vector<uint8_t> extract(const std::string& name) const;

private:
    std::vector<uint8_t> data_;
};
