#pragma once
#include "CameraGestures/Types.h"
#include <vector>
#include <array>

/* Constants matching preprocessor.js wire contract. */
inline constexpr int CG_TARGET_FRAMES      = 60;
inline constexpr int CG_LANDMARKS          = 21;
inline constexpr int CG_COORDS_PER_FRAME   = CG_LANDMARKS * 3;          // 63
inline constexpr int CG_POSE_EXTRAS        = 20;
inline constexpr int CG_POSE_VECTOR_SIZE   = CG_COORDS_PER_FRAME + CG_POSE_EXTRAS; // 83
inline constexpr int CG_FEATURES_PER_FRAME = CG_POSE_VECTOR_SIZE + CG_COORDS_PER_FRAME; // 146
inline constexpr int CG_SUMMARY_FEATURES   = 296;

/*
 * Pure C++ port of preprocessor.js.
 * All methods are const and stateless except for geom_coef_.
 * Thread-safe after construction provided geom_coef_ is set before concurrent use.
 */
class FeaturePreprocessor {
public:
    FeaturePreprocessor() = default;

    void  setGeomCoef(float c) { geom_coef_ = c; }
    float geomCoef()     const { return geom_coef_; }

    /* Flat array of TARGET_FRAMES * FEATURES_PER_FRAME = 8760 floats.
     * Matches featureMatrix() in preprocessor.js. */
    std::vector<float> featureMatrix(cg_handfilm_ref film) const;

    /* 296-element statistical summary vector (summaryFeatures() in JS). */
    std::vector<float> summaryFeatures(cg_handfilm_ref film) const;

    /* 83-float pose vector for a single frame (poseVector() in JS). */
    std::vector<float> poseVector(const cg_handshot& shot) const;

private:
    float geom_coef_ = 1.0f;

    using Vec3     = std::array<float, 3>;
    using Landmark = std::array<Vec3, CG_LANDMARKS>;

    /* Filter absent and zero-landmark frames — mirrors nonAbsentFrames() in JS. */
    std::vector<cg_handshot> nonAbsentFrames(cg_handfilm_ref film) const;

    /* Steps 1-4 normalisation; returns 21×3 palm-local coords.
     * is_left: handedness flip (mirror x). */
    Landmark normaliseFrame(const cg_point3d* landmarks, bool is_left) const;

    /* 20 geometric extras: 10 pairwise distances + 5 flexion angles + 5 palm distances. */
    std::array<float, CG_POSE_EXTRAS> poseExtras(const Landmark& rel) const;
};
