/*
 * FeaturePreprocessor.cpp — C++ port of preprocessor.js.
 *
 * All logic must produce bit-identical float results to the JS implementation
 * (within IEEE 754 float32 rounding). The parity test in test_gesture_model.cpp
 * verifies this against Python-generated reference vectors.
 *
 * Algorithm reference: CameraGestures-server/server/ml/preprocessor.js
 */

#include "FeaturePreprocessor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

static inline float dot3(const std::array<float,3>& a, const std::array<float,3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static inline float norm3(const std::array<float,3>& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/* -------------------------------------------------------------------------
 * nonAbsentFrames — filter absent + zero-landmark frames
 * ---------------------------------------------------------------------- */

std::vector<cg_handshot> FeaturePreprocessor::nonAbsentFrames(cg_handfilm_ref film) const {
    if (!film) return {};
    size_t count = cg_handfilm_shot_count(film);
    std::vector<cg_handshot> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        cg_handshot s{};
        if (!cg_handfilm_get_shot(film, i, &s)) continue;
        if (s.is_absent) continue;
        const auto& lm9   = s.landmarks[9];
        const auto& wrist = s.landmarks[0];
        float dx = lm9.x - wrist.x, dy = lm9.y - wrist.y, dz = lm9.z - wrist.z;
        if (dx*dx + dy*dy + dz*dz < 1e-12f) continue;  // zero-landmark guard
        result.push_back(s);
    }
    return result;
}

/* -------------------------------------------------------------------------
 * normaliseFrame — 4-step pipeline (mirror JS _normaliseFrame exactly)
 * ---------------------------------------------------------------------- */

FeaturePreprocessor::Landmark
FeaturePreprocessor::normaliseFrame(const cg_point3d* lm, bool is_left) const {
    const float xSign = is_left ? -1.0f : 1.0f;
    const float wx = lm[0].x, wy = lm[0].y, wz = lm[0].z;

    // Step 1+2: handedness flip + wrist-relative
    Landmark rel{};
    for (int j = 0; j < CG_LANDMARKS; ++j) {
        rel[j] = { (lm[j].x - wx) * xSign,
                    lm[j].y - wy,
                    lm[j].z - wz };
    }

    // Step 3: scale by |wrist→middle_MCP (lm9)|
    float scale = norm3(rel[9]);
    if (scale < 1e-6f) scale = 1.0f;
    for (auto& p : rel) { p[0] /= scale; p[1] /= scale; p[2] /= scale; }

    // Step 4: rotate into palm-local frame
    // up = normalised wrist→lm9 (already unit after step 3, but lm9 IS the up vector)
    std::array<float,3> up = rel[9];  // unit by construction after scale step

    // right = (lm17 - lm5) projected perpendicular to up
    std::array<float,3> rApprox = {
        rel[17][0] - rel[5][0],
        rel[17][1] - rel[5][1],
        rel[17][2] - rel[5][2]
    };
    float dotRU = dot3(rApprox, up);
    std::array<float,3> right = {
        rApprox[0] - dotRU * up[0],
        rApprox[1] - dotRU * up[1],
        rApprox[2] - dotRU * up[2]
    };
    float rMag = norm3(right);
    if (rMag < 1e-6f) {
        right = {1.0f, 0.0f, 0.0f};  // degenerate fallback
    } else {
        right[0] /= rMag; right[1] /= rMag; right[2] /= rMag;
    }

    // forward = up × right
    std::array<float,3> fwd = {
        up[1]*right[2] - up[2]*right[1],
        up[2]*right[0] - up[0]*right[2],
        up[0]*right[1] - up[1]*right[0]
    };

    // Rotate each landmark into (right, up, forward) basis
    for (auto& p : rel) {
        float nx = p[0]*right[0] + p[1]*right[1] + p[2]*right[2];
        float ny = p[0]*up[0]    + p[1]*up[1]    + p[2]*up[2];
        float nz = p[0]*fwd[0]   + p[1]*fwd[1]   + p[2]*fwd[2];
        p = {nx, ny, nz};
    }

    return rel;
}

/* -------------------------------------------------------------------------
 * poseExtras — 20 engineered geometric features
 * ---------------------------------------------------------------------- */

std::array<float, CG_POSE_EXTRAS>
FeaturePreprocessor::poseExtras(const Landmark& rel) const {
    std::array<float, CG_POSE_EXTRAS> out{};
    int idx = 0;

    // Block A: pairwise fingertip distances (10 floats)
    // tips: thumb=4, index=8, middle=12, ring=16, pinky=20
    static constexpr int tips[5] = {4, 8, 12, 16, 20};
    for (int i = 0; i < 5; ++i) {
        for (int j = i+1; j < 5; ++j) {
            float dx = rel[tips[i]][0] - rel[tips[j]][0];
            float dy = rel[tips[i]][1] - rel[tips[j]][1];
            float dz = rel[tips[i]][2] - rel[tips[j]][2];
            out[idx++] = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }

    // Block B: PIP/IP flexion angles (5 floats, radians)
    // angle = acos(clamp(dot(normalize(PIP-MCP), normalize(tip-PIP)), -1, 1))
    static constexpr int fingers[5][3] = {
        {5,  6,  8 },   // index
        {9,  10, 12},   // middle
        {13, 14, 16},   // ring
        {17, 18, 20},   // pinky
        {2,  3,  4 }    // thumb
    };
    for (int k = 0; k < 5; ++k) {
        const auto& mcp = rel[fingers[k][0]];
        const auto& pip = rel[fingers[k][1]];
        const auto& tip = rel[fingers[k][2]];
        float v1x = pip[0]-mcp[0], v1y = pip[1]-mcp[1], v1z = pip[2]-mcp[2];
        float m1 = std::sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
        if (m1 < 1e-9f) { out[idx++] = 0.0f; continue; }
        v1x /= m1; v1y /= m1; v1z /= m1;
        float v2x = tip[0]-pip[0], v2y = tip[1]-pip[1], v2z = tip[2]-pip[2];
        float m2 = std::sqrt(v2x*v2x + v2y*v2y + v2z*v2z);
        if (m2 < 1e-9f) { out[idx++] = 0.0f; continue; }
        v2x /= m2; v2y /= m2; v2z /= m2;
        float d = v1x*v2x + v1y*v2y + v1z*v2z;
        d = d < -1.0f ? -1.0f : (d > 1.0f ? 1.0f : d);
        out[idx++] = std::acos(d);
    }

    // Block C: fingertip-to-palm-centre distances (5 floats)
    // palm centre = mean of wrist(0), index_MCP(5), middle_MCP(9), ring_MCP(13), pinky_MCP(17)
    static constexpr int palmIdx[5] = {0, 5, 9, 13, 17};
    float pcx = 0, pcy = 0, pcz = 0;
    for (int i = 0; i < 5; ++i) {
        pcx += rel[palmIdx[i]][0];
        pcy += rel[palmIdx[i]][1];
        pcz += rel[palmIdx[i]][2];
    }
    pcx /= 5.0f; pcy /= 5.0f; pcz /= 5.0f;
    for (int i = 0; i < 5; ++i) {
        float dx = rel[tips[i]][0] - pcx;
        float dy = rel[tips[i]][1] - pcy;
        float dz = rel[tips[i]][2] - pcz;
        out[idx++] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    // Apply geom_coef if not 1.0
    if (geom_coef_ != 1.0f) {
        for (auto& v : out) v *= geom_coef_;
    }

    return out;
}

/* -------------------------------------------------------------------------
 * poseVector — 83-float per-frame vector (for Phase 2 HoldDetector)
 * ---------------------------------------------------------------------- */

std::vector<float> FeaturePreprocessor::poseVector(const cg_handshot& shot) const {
    bool is_left = (shot.handedness == CG_HAND_LEFT);
    Landmark rel = normaliseFrame(shot.landmarks, is_left);
    std::vector<float> row;
    row.reserve(CG_POSE_VECTOR_SIZE);
    for (const auto& p : rel) { row.push_back(p[0]); row.push_back(p[1]); row.push_back(p[2]); }
    auto extras = poseExtras(rel);
    for (float v : extras) row.push_back(v);
    return row;  // 83 floats
}

/* -------------------------------------------------------------------------
 * featureMatrix — TARGET_FRAMES × FEATURES_PER_FRAME flat array
 * ---------------------------------------------------------------------- */

std::vector<float> FeaturePreprocessor::featureMatrix(cg_handfilm_ref film) const {
    auto frames = nonAbsentFrames(film);
    int n = static_cast<int>(frames.size());

    // Normalise all present frames
    std::vector<Landmark> normalised(n);
    for (int i = 0; i < n; ++i) {
        bool is_left = (frames[i].handedness == CG_HAND_LEFT);
        normalised[i] = normaliseFrame(frames[i].landmarks, is_left);
    }

    // Velocity: frame-to-frame delta of landmark coords only (63 floats, zero for first)
    std::vector<std::array<Vec3, CG_LANDMARKS>> velocity(n);
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            for (auto& v : velocity[i]) v = {0.0f, 0.0f, 0.0f};
        } else {
            for (int j = 0; j < CG_LANDMARKS; ++j) {
                velocity[i][j] = {
                    normalised[i][j][0] - normalised[i-1][j][0],
                    normalised[i][j][1] - normalised[i-1][j][1],
                    normalised[i][j][2] - normalised[i-1][j][2]
                };
            }
        }
    }

    // Build rows: [coords(63), extras(20), velocity(63)] = 146 floats each
    std::vector<std::array<float, CG_FEATURES_PER_FRAME>> rows(n);
    for (int i = 0; i < n; ++i) {
        int col = 0;
        for (int j = 0; j < CG_LANDMARKS; ++j) {
            rows[i][col++] = normalised[i][j][0];
            rows[i][col++] = normalised[i][j][1];
            rows[i][col++] = normalised[i][j][2];
        }
        auto extras = poseExtras(normalised[i]);
        for (float v : extras) rows[i][col++] = v;
        for (int j = 0; j < CG_LANDMARKS; ++j) {
            rows[i][col++] = velocity[i][j][0];
            rows[i][col++] = velocity[i][j][1];
            rows[i][col++] = velocity[i][j][2];
        }
    }

    // Build output: pad (append zeros) or trim (keep last TARGET_FRAMES)
    std::vector<float> flat(CG_TARGET_FRAMES * CG_FEATURES_PER_FRAME, 0.0f);
    if (n >= CG_TARGET_FRAMES) {
        // Keep last TARGET_FRAMES rows
        int start = n - CG_TARGET_FRAMES;
        for (int i = 0; i < CG_TARGET_FRAMES; ++i) {
            std::copy(rows[start+i].begin(), rows[start+i].end(),
                      flat.begin() + i * CG_FEATURES_PER_FRAME);
        }
    } else {
        // Copy all n rows starting at row 0; remaining rows stay zero (prepended zeros)
        for (int i = 0; i < n; ++i) {
            std::copy(rows[i].begin(), rows[i].end(),
                      flat.begin() + i * CG_FEATURES_PER_FRAME);
        }
    }
    return flat;
}

/* -------------------------------------------------------------------------
 * summaryFeatures — 296-element statistical summary
 * ---------------------------------------------------------------------- */

std::vector<float> FeaturePreprocessor::summaryFeatures(cg_handfilm_ref film) const {
    auto flat = featureMatrix(film);
    auto frames = nonAbsentFrames(film);
    int realCount = std::min(static_cast<int>(frames.size()), CG_TARGET_FRAMES);

    // Split flat into pose (83/frame) and velocity (63/frame) columns
    // poses[i] and vels[i] are views into the flat array
    auto getPose = [&](int i) -> const float* {
        return flat.data() + i * CG_FEATURES_PER_FRAME;
    };
    auto getVel = [&](int i) -> const float* {
        return flat.data() + i * CG_FEATURES_PER_FRAME + CG_POSE_VECTOR_SIZE;
    };

    // Column-wise mean and std over realCount frames
    auto colMean = [&](auto getter, int ncols) {
        std::vector<float> mean(ncols, 0.0f);
        if (realCount == 0) return mean;
        for (int i = 0; i < realCount; ++i) {
            const float* row = getter(i);
            for (int j = 0; j < ncols; ++j) mean[j] += row[j];
        }
        for (auto& v : mean) v /= realCount;
        return mean;
    };
    auto colStd = [&](auto getter, const std::vector<float>& mean, int ncols) {
        std::vector<float> std_v(ncols, 0.0f);
        if (realCount == 0) return std_v;
        for (int i = 0; i < realCount; ++i) {
            const float* row = getter(i);
            for (int j = 0; j < ncols; ++j) {
                float d = row[j] - mean[j];
                std_v[j] += d * d;
            }
        }
        for (auto& v : std_v) v = std::sqrt(v / realCount);
        return std_v;
    };

    auto poseMean = colMean(getPose, CG_POSE_VECTOR_SIZE);
    auto poseStd  = colStd(getPose, poseMean, CG_POSE_VECTOR_SIZE);
    auto velMean  = colMean(getVel,  CG_COORDS_PER_FRAME);
    auto velStd   = colStd(getVel,   velMean, CG_COORDS_PER_FRAME);

    // Net raw wrist displacement (last - first frame, before normalisation, x flipped for left)
    float disp[3] = {0.0f, 0.0f, 0.0f};
    if (static_cast<int>(frames.size()) >= 2) {
        const auto& first = frames.front();
        const auto& last  = frames.back();
        float firstSign = (first.handedness == CG_HAND_LEFT) ? -1.0f : 1.0f;
        float lastSign  = (last.handedness  == CG_HAND_LEFT) ? -1.0f : 1.0f;
        disp[0] = last.landmarks[0].x * lastSign  - first.landmarks[0].x * firstSign;
        disp[1] = last.landmarks[0].y             - first.landmarks[0].y;
        disp[2] = last.landmarks[0].z             - first.landmarks[0].z;
    }

    // Dominant motion axis: max |mean velocity| across x/y/z averaged over all landmarks
    float axisSum[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < realCount; ++i) {
        const float* v = getVel(i);
        for (int lm = 0; lm < CG_LANDMARKS; ++lm) {
            axisSum[0] += v[lm*3 + 0];
            axisSum[1] += v[lm*3 + 1];
            axisSum[2] += v[lm*3 + 2];
        }
    }
    float total = static_cast<float>((realCount == 0 ? 1 : realCount) * CG_LANDMARKS);
    float dominant = std::max({std::abs(axisSum[0]/total),
                                std::abs(axisSum[1]/total),
                                std::abs(axisSum[2]/total)});

    // Assemble: poseMean(83) + poseStd(83) + velMean(63) + velStd(63) + disp(3) + dominant(1) = 296
    std::vector<float> result;
    result.reserve(CG_SUMMARY_FEATURES);
    result.insert(result.end(), poseMean.begin(), poseMean.end());
    result.insert(result.end(), poseStd.begin(),  poseStd.end());
    result.insert(result.end(), velMean.begin(),  velMean.end());
    result.insert(result.end(), velStd.begin(),   velStd.end());
    result.push_back(disp[0]);
    result.push_back(disp[1]);
    result.push_back(disp[2]);
    result.push_back(dominant);

    return result;  // 296 floats
}
