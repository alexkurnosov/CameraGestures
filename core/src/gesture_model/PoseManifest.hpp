#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <stdexcept>

/* Mirror of iOS GestureModel/PoseManifest.swift */

enum class CgClusterKind { regular, idle, unconfirmed };

struct CgPoseCluster {
    std::string         label;
    CgClusterKind       kind         = CgClusterKind::unconfirmed;
    bool                suspected_idle = false;
    int                 n_samples    = 0;
    std::vector<float>  centroid;
};

struct CgPoseManifestParameters {
    std::optional<float>  t_hold;
    std::optional<int>    k_hold_frames;
    std::optional<int>    smooth_k;
    std::optional<float>  edge_trim_fraction;
    std::optional<double> t_commit_ms;
    std::optional<float>  epsilon;
    std::optional<float>  tau_pose_confidence;
};

struct CgPoseManifest {
    int version = 0;
    std::unordered_map<std::string, CgPoseCluster>          pose_clusters;
    std::vector<int>                                         idle_poses;
    std::unordered_map<std::string, std::vector<std::vector<int>>> gesture_templates;
    std::optional<CgPoseManifestParameters>                  parameters;

    CgClusterKind clusterKind(int id) const;
    std::string   clusterLabel(int id) const;

    /* Sorted cluster IDs (matches poseClusterIds logic in iOS GestureModel). */
    std::vector<std::string> sortedClusterIds() const;

    /* Load from a pose_manifest.json file. Throws std::runtime_error on failure. */
    static CgPoseManifest loadFromFile(const std::string& path);
};
