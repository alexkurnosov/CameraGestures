#include "GestureModel.hpp"
#include <algorithm>
#include <cstring>
#include <ctime>

/* -------------------------------------------------------------------------
 * load — Phase 3 gesture MLP
 * ---------------------------------------------------------------------- */

bool GestureModel::load(const std::string& tflite_path, std::vector<std::string> gesture_ids) {
    if (!backend_.load(tflite_path)) return false;
    int expected = backend_.inputDim();
    if (expected > 0 && expected != CG_SUMMARY_FEATURES) return false;  // version mismatch
    gesture_ids_ = std::move(gesture_ids);
    return true;
}

/* -------------------------------------------------------------------------
 * loadPoseModel — Phase 2 pose MLP
 * ---------------------------------------------------------------------- */

bool GestureModel::loadPoseModel(const std::string& tflite_path,
                                  const std::string& manifest_path) {
    CgPoseManifest m;
    try { m = CgPoseManifest::loadFromFile(manifest_path); }
    catch (...) { return false; }

    if (!pose_backend_.load(tflite_path)) return false;
    int expected = pose_backend_.inputDim();
    if (expected > 0 && expected != CG_POSE_VECTOR_SIZE) return false;

    pose_manifest_    = std::move(m);
    pose_cluster_ids_ = pose_manifest_->sortedClusterIds();
    return true;
}

/* -------------------------------------------------------------------------
 * runPhase3 — shared inference helper
 * ---------------------------------------------------------------------- */

bool GestureModel::runPhase3(cg_handfilm_ref film, std::vector<float>& probs_out) {
    if (!backend_.isLoaded() || gesture_ids_.empty()) return false;
    if (cg_handfilm_shot_count(film) == 0) return false;

    auto features = preprocessor_.summaryFeatures(film);
    if (!backend_.run(features, probs_out)) return false;
    if (static_cast<int>(probs_out.size()) != static_cast<int>(gesture_ids_.size())) return false;

    return true;
}

/* -------------------------------------------------------------------------
 * classify — top-1 with threshold
 * ---------------------------------------------------------------------- */

int GestureModel::classify(cg_handfilm_ref film, float threshold, cg_gesture_prediction* out) {
    return classifyTopK(film, 1, threshold, out, 1);
}

/* -------------------------------------------------------------------------
 * classifyTopK
 * ---------------------------------------------------------------------- */

int GestureModel::classifyTopK(cg_handfilm_ref film, int k, float threshold,
                                 cg_gesture_prediction* out, int max_out) {
    std::vector<float> probs;
    if (!runPhase3(film, probs)) return 0;

    double ts = static_cast<double>(std::time(nullptr));

    // Collect candidates above threshold, excluding _none
    std::vector<std::pair<float,int>> candidates;
    candidates.reserve(gesture_ids_.size());
    for (int i = 0; i < static_cast<int>(gesture_ids_.size()); ++i) {
        if (gesture_ids_[i] == NONE_GESTURE_ID) continue;
        if (probs[i] < threshold) continue;
        candidates.push_back({probs[i], i});
    }

    // Sort descending by confidence
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int limit  = std::min({k, max_out, static_cast<int>(candidates.size())});
    for (int n = 0; n < limit; ++n) {
        auto [conf, idx] = candidates[n];
        cg_gesture_prediction& p = out[n];
        std::strncpy(p.gesture_id,   gesture_ids_[idx].c_str(), sizeof(p.gesture_id)-1);
        std::strncpy(p.gesture_name, gesture_ids_[idx].c_str(), sizeof(p.gesture_name)-1);
        p.gesture_id[sizeof(p.gesture_id)-1]     = '\0';
        p.gesture_name[sizeof(p.gesture_name)-1] = '\0';
        p.confidence = conf;
        p.timestamp  = ts;
    }
    return limit;
}

/* -------------------------------------------------------------------------
 * classifyRestricted — masked argmax
 * ---------------------------------------------------------------------- */

int GestureModel::classifyRestricted(cg_handfilm_ref film,
                                      const char* const* candidate_ids, int n_candidates,
                                      float threshold, cg_gesture_prediction* out) {
    std::vector<float> probs;
    if (!runPhase3(film, probs)) return 0;
    if (n_candidates <= 0 || !candidate_ids) return 0;

    float bestProb = -1.0f;
    int   bestIdx  = -1;
    for (int i = 0; i < static_cast<int>(gesture_ids_.size()); ++i) {
        if (gesture_ids_[i] == NONE_GESTURE_ID) continue;
        // Check if this gesture is in the candidate set
        bool inSet = false;
        for (int c = 0; c < n_candidates; ++c) {
            if (candidate_ids[c] && gesture_ids_[i] == candidate_ids[c]) {
                inSet = true; break;
            }
        }
        if (!inSet) continue;
        if (probs[i] > bestProb) { bestProb = probs[i]; bestIdx = i; }
    }

    if (bestIdx < 0 || bestProb < threshold) return 0;

    double ts = static_cast<double>(std::time(nullptr));
    std::strncpy(out->gesture_id,   gesture_ids_[bestIdx].c_str(), sizeof(out->gesture_id)-1);
    std::strncpy(out->gesture_name, gesture_ids_[bestIdx].c_str(), sizeof(out->gesture_name)-1);
    out->gesture_id[sizeof(out->gesture_id)-1]     = '\0';
    out->gesture_name[sizeof(out->gesture_name)-1] = '\0';
    out->confidence = bestProb;
    out->timestamp  = ts;
    return 1;
}

/* -------------------------------------------------------------------------
 * predictPose — Phase 2 pose cluster classification
 * ---------------------------------------------------------------------- */

int GestureModel::predictPose(const cg_handshot& shot,
                               int* pose_id_out, float* confidence_out) {
    if (!pose_backend_.isLoaded() || !pose_manifest_.has_value()) return 0;
    if (pose_cluster_ids_.empty()) return 0;

    auto pv = preprocessor_.poseVector(shot);
    std::vector<float> probs;
    if (!pose_backend_.run(pv, probs)) return 0;
    if (probs.size() != pose_cluster_ids_.size()) return 0;

    auto it = std::max_element(probs.begin(), probs.end());
    int  maxIdx   = static_cast<int>(std::distance(probs.begin(), it));
    float maxProb = *it;

    int clusterId = 0;
    try { clusterId = std::stoi(pose_cluster_ids_[maxIdx]); } catch(...) { clusterId = maxIdx; }

    if (pose_id_out)     *pose_id_out     = clusterId;
    if (confidence_out)  *confidence_out  = maxProb;
    return 1;
}

/* =========================================================================
 * C ABI
 * ====================================================================== */

struct cg_gesture_model_s {
    GestureModel impl;
};

extern "C" {

cg_gesture_model_ref cg_gesture_model_load(const char* tflite_path,
                                             const char* registry_path) {
    if (!tflite_path || !registry_path) return nullptr;

    // Build gesture_ids from the registry: load all IDs, add _none, sort alphabetically.
    cg_registry_ref reg = cg_registry_create(registry_path);
    if (!reg) return nullptr;

    std::vector<std::string> ids;
    size_t count = cg_registry_count(reg);
    ids.reserve(count + 1);
    for (size_t i = 0; i < count; ++i) {
        cg_gesture_definition def{};
        if (cg_registry_get(reg, i, &def)) {
            ids.emplace_back(def.id);
        }
    }
    cg_registry_destroy(reg);

    ids.emplace_back("_none");
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    auto* ref = new cg_gesture_model_s{};
    if (!ref->impl.load(tflite_path, std::move(ids))) {
        delete ref;
        return nullptr;
    }
    return ref;
}

void cg_gesture_model_destroy(cg_gesture_model_ref ref) {
    delete ref;
}

int cg_gesture_model_load_pose(cg_gesture_model_ref ref,
                                 const char* tflite_path,
                                 const char* manifest_path) {
    if (!ref || !tflite_path || !manifest_path) return 0;
    return ref->impl.loadPoseModel(tflite_path, manifest_path) ? 1 : 0;
}

int cg_gesture_model_classify(cg_gesture_model_ref ref,
                                cg_handfilm_ref film,
                                float threshold,
                                cg_gesture_prediction* out) {
    if (!ref || !film || !out) return 0;
    return ref->impl.classify(film, threshold, out);
}

int cg_gesture_model_classify_topk(cg_gesture_model_ref ref,
                                     cg_handfilm_ref film,
                                     int k,
                                     float threshold,
                                     cg_gesture_prediction* out,
                                     int max_out) {
    if (!ref || !film || !out || k <= 0 || max_out <= 0) return 0;
    return ref->impl.classifyTopK(film, k, threshold, out, max_out);
}

int cg_gesture_model_classify_restricted(cg_gesture_model_ref ref,
                                           cg_handfilm_ref film,
                                           const char* const* candidate_ids,
                                           int n_candidates,
                                           float threshold,
                                           cg_gesture_prediction* out) {
    if (!ref || !film || !out || !candidate_ids || n_candidates <= 0) return 0;
    return ref->impl.classifyRestricted(film, candidate_ids, n_candidates, threshold, out);
}

int cg_gesture_model_predict_pose(cg_gesture_model_ref ref,
                                    const cg_handshot* shot,
                                    int* pose_id_out,
                                    float* confidence_out) {
    if (!ref || !shot) return 0;
    return ref->impl.predictPose(*shot, pose_id_out, confidence_out);
}

void cg_gesture_model_set_geom_coef(cg_gesture_model_ref ref, float coef) {
    if (ref) ref->impl.setGeomCoef(coef);
}

int cg_gesture_model_gesture_count(cg_gesture_model_ref ref) {
    if (!ref) return 0;
    return static_cast<int>(ref->impl.gestureIds().size());
}

} // extern "C"
