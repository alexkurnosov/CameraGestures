#include "CameraGestures/Types.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <filesystem>

using json = nlohmann::json;

/* -------------------------------------------------------------------------
 * Slug helper — mirrors GestureRegistry.slug(from:) in Swift
 * "Thumbs Up" → "thumbs_up"
 * ---------------------------------------------------------------------- */

static std::string make_slug(const std::string& name) {
    std::string slug;
    for (char c : name) {
        if (std::isalpha((unsigned char)c) || std::isdigit((unsigned char)c)) {
            slug += std::tolower((unsigned char)c);
        } else if (c == ' ') {
            slug += '_';
        }
    }
    return slug;
}

static void copy_str(char* dst, size_t dst_size, const std::string& src) {
    std::strncpy(dst, src.c_str(), dst_size - 1);
    dst[dst_size - 1] = '\0';
}

/* -------------------------------------------------------------------------
 * Internal registry struct
 * ---------------------------------------------------------------------- */

struct GestureEntry {
    std::string id;
    std::string name;
    std::string description;
};

struct cg_gesture_registry_s {
    std::string file_path;
    std::vector<GestureEntry> entries;

    explicit cg_gesture_registry_s(const char* path) : file_path(path) {
        load();
    }

    void load() {
        std::ifstream f(file_path);
        if (!f.good()) return;
        try {
            auto arr = json::parse(f);
            for (const auto& obj : arr) {
                GestureEntry e;
                e.id          = obj.value("id",          "");
                e.name        = obj.value("name",        "");
                e.description = obj.value("description", "");
                if (!e.id.empty()) entries.push_back(std::move(e));
            }
        } catch (...) {}
    }

    bool save() const {
        try {
            std::filesystem::create_directories(
                std::filesystem::path(file_path).parent_path());
            json arr = json::array();
            for (const auto& e : entries) {
                arr.push_back({{"id", e.id}, {"name", e.name},
                               {"description", e.description}});
            }
            std::ofstream f(file_path);
            if (!f.good()) return false;
            f << arr.dump(2);
            return true;
        } catch (...) {
            return false;
        }
    }

    void fill(size_t index, cg_gesture_definition* out) const {
        const auto& e = entries[index];
        copy_str(out->id,          sizeof(out->id),          e.id);
        copy_str(out->name,        sizeof(out->name),        e.name);
        copy_str(out->description, sizeof(out->description), e.description);
    }
};

/* -------------------------------------------------------------------------
 * C ABI
 * ---------------------------------------------------------------------- */

cg_registry_ref cg_registry_create(const char* file_path) {
    return new cg_gesture_registry_s(file_path);
}

void cg_registry_destroy(cg_registry_ref registry) {
    delete registry;
}

size_t cg_registry_count(cg_registry_ref registry) {
    return registry ? registry->entries.size() : 0;
}

int cg_registry_get(cg_registry_ref registry, size_t index,
                    cg_gesture_definition* out) {
    if (!registry || index >= registry->entries.size() || !out) return 0;
    registry->fill(index, out);
    return 1;
}

int cg_registry_find_by_id(cg_registry_ref registry, const char* id,
                            cg_gesture_definition* out) {
    if (!registry || !id) return 0;
    for (size_t i = 0; i < registry->entries.size(); ++i) {
        if (registry->entries[i].id == id) {
            if (out) registry->fill(i, out);
            return 1;
        }
    }
    return 0;
}

int cg_registry_add(cg_registry_ref registry,
                    const char* name, const char* description,
                    cg_gesture_definition* out) {
    if (!registry || !name) return 0;
    std::string slug = make_slug(std::string(name));
    if (slug.empty()) return 0;
    for (const auto& e : registry->entries) {
        if (e.id == slug) return 0;
    }
    GestureEntry entry{slug, std::string(name),
                       description ? std::string(description) : ""};
    registry->entries.push_back(entry);
    registry->save();
    if (out) registry->fill(registry->entries.size() - 1, out);
    return 1;
}

int cg_registry_remove(cg_registry_ref registry, const char* id) {
    if (!registry || !id) return 0;
    auto& v = registry->entries;
    auto it = std::find_if(v.begin(), v.end(),
                           [&](const GestureEntry& e){ return e.id == id; });
    if (it == v.end()) return 0;
    v.erase(it);
    registry->save();
    return 1;
}

int cg_registry_save(cg_registry_ref registry) {
    return registry ? (registry->save() ? 1 : 0) : 0;
}
