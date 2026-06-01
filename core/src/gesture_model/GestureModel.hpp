#pragma once
#include "CameraGestures/Types.h"
#include "CameraGestures/GestureModel.h"
#include "FeaturePreprocessor.hpp"
#include "TFLiteBackend.hpp"
#include "PoseManifest.hpp"
#include <string>
#include <vector>
#include <optional>

class GestureModel {
public:
    GestureModel()  = default;
    ~GestureModel() = default;

    /* Load Phase-3 gesture classifier.
     * gesture_ids must be in the same order as the model's output classes
     * (i.e. sorted alphabetically, matching the server trainer's output). */
    bool load(const std::string& tflite_path, std::vector<std::string> gesture_ids);

    /* Load Phase-2 pose MLP and its manifest JSON. */
    bool loadPoseModel(const std::string& tflite_path, const std::string& manifest_path);

    bool isLoaded()     const { return backend_.isLoaded(); }
    bool isPoseLoaded() const { return pose_backend_.isLoaded() && pose_manifest_.has_value(); }

    void setGeomCoef(float c) { preprocessor_.setGeomCoef(c); }

    const std::vector<std::string>& gestureIds() const { return gesture_ids_; }
    const CgPoseManifest*           poseManifest() const {
        return pose_manifest_.has_value() ? &pose_manifest_.value() : nullptr;
    }

    /* Phase-3: classify a full HandFilm.
     * Returns 1 and fills *out on success; 0 otherwise. */
    int classify(cg_handfilm_ref film, float threshold, cg_gesture_prediction* out);

    /* Top-k variant; returns count written (≤ k, ≤ max_out). */
    int classifyTopK(cg_handfilm_ref film, int k, float threshold,
                     cg_gesture_prediction* out, int max_out);

    /* Masked argmax: only consider gestures in candidate_ids (null-terminated array). */
    int classifyRestricted(cg_handfilm_ref film,
                           const char* const* candidate_ids, int n_candidates,
                           float threshold, cg_gesture_prediction* out);

    /* Phase-2: predict pose cluster for a single frame.
     * pose_id_out: cluster integer ID. Returns 1 on success. */
    int predictPose(const cg_handshot& shot, int* pose_id_out, float* confidence_out);

    /* Like predictPose but also fills scores_out[0..n) with softmax probabilities
     * for every pose class in class_labels order.
     * scores_out/n_scores_out may be null (behaves like predictPose). */
    int predictPoseAllScores(const cg_handshot& shot,
                              int*   pose_id_out,
                              float* confidence_out,
                              float* scores_out,
                              int    scores_capacity,
                              int*   n_scores_out);

    int poseClassCount() const { return static_cast<int>(pose_cluster_ids_.size()); }

private:
    FeaturePreprocessor              preprocessor_;
    TFLiteBackend                    backend_;
    TFLiteBackend                    pose_backend_;
    std::vector<std::string>         gesture_ids_;
    std::vector<std::string>         pose_cluster_ids_;  // sorted numerically
    std::optional<CgPoseManifest>    pose_manifest_;

    static constexpr const char* NONE_GESTURE_ID = "_none";

    /* Shared inference helper: runs the Phase-3 model and returns softmax probabilities. */
    bool runPhase3(cg_handfilm_ref film, std::vector<float>& probs_out);
};
