#include "PoseManifest.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>

using json = nlohmann::json;

static CgClusterKind parseKind(const std::string& s) {
    if (s == "regular")    return CgClusterKind::regular;
    if (s == "idle")       return CgClusterKind::idle;
    return CgClusterKind::unconfirmed;
}

CgClusterKind CgPoseManifest::clusterKind(int id) const {
    auto it = pose_clusters.find(std::to_string(id));
    if (it == pose_clusters.end()) return CgClusterKind::unconfirmed;
    return it->second.kind;
}

std::string CgPoseManifest::clusterLabel(int id) const {
    auto it = pose_clusters.find(std::to_string(id));
    if (it == pose_clusters.end()) return "pose_" + std::to_string(id);
    return it->second.label;
}

std::vector<std::string> CgPoseManifest::sortedClusterIds() const {
    std::vector<std::string> ids;
    ids.reserve(pose_clusters.size());
    for (const auto& kv : pose_clusters) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) {
        int ia = 0, ib = 0;
        try { ia = std::stoi(a); } catch(...) {}
        try { ib = std::stoi(b); } catch(...) {}
        return ia < ib;
    });
    return ids;
}

CgPoseManifest CgPoseManifest::loadFromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("PoseManifest: cannot open " + path);
    json j;
    f >> j;

    CgPoseManifest m;
    m.version = j.value("version", 0);

    if (j.contains("pose_clusters")) {
        for (const auto& [key, val] : j["pose_clusters"].items()) {
            CgPoseCluster c;
            c.label          = val.value("label", "");
            c.kind           = parseKind(val.value("kind", "unconfirmed"));
            c.suspected_idle = val.value("suspected_idle", false);
            c.n_samples      = val.value("n_samples", 0);
            if (val.contains("centroid")) {
                c.centroid = val["centroid"].get<std::vector<float>>();
            }
            m.pose_clusters[key] = std::move(c);
        }
    }

    if (j.contains("idle_poses")) {
        m.idle_poses = j["idle_poses"].get<std::vector<int>>();
    }

    if (j.contains("gesture_templates")) {
        for (const auto& [key, val] : j["gesture_templates"].items()) {
            m.gesture_templates[key] = val.get<std::vector<std::vector<int>>>();
        }
    }

    if (j.contains("parameters") && !j["parameters"].is_null()) {
        const auto& p = j["parameters"];
        CgPoseManifestParameters params;
        if (p.contains("t_hold"))             params.t_hold             = p["t_hold"].get<float>();
        if (p.contains("k_hold_frames"))      params.k_hold_frames      = p["k_hold_frames"].get<int>();
        if (p.contains("smooth_k"))           params.smooth_k           = p["smooth_k"].get<int>();
        if (p.contains("edge_trim_fraction")) params.edge_trim_fraction = p["edge_trim_fraction"].get<float>();
        if (p.contains("t_commit_ms"))        params.t_commit_ms        = p["t_commit_ms"].get<double>();
        if (p.contains("epsilon"))            params.epsilon            = p["epsilon"].get<float>();
        if (p.contains("tau_pose_confidence"))params.tau_pose_confidence = p["tau_pose_confidence"].get<float>();
        m.parameters = params;
    }

    return m;
}
