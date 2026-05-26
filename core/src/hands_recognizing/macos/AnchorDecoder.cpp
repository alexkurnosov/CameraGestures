#include "AnchorDecoder.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

// ---- Anchor generation ------------------------------------------------------
// MediaPipe SsdAnchorsCalculator for palm_detection_full_192x192:
//   input_size: 192, strides: [8,16,16,16], aspect_ratios: [1.0],
//   fixed_anchor_size: true, interpolated_scale_aspect_ratio: 1.0
//   → 2 anchors per cell per layer, 2016 total.

std::vector<Anchor> generatePalmAnchors() {
    static const int   input_size  = 192;
    static const int   strides[]   = {8, 16, 16, 16};
    static const int   num_layers  = 4;
    static const int   anchors_per_cell = 2; // 1 aspect_ratio + 1 interpolated

    std::vector<Anchor> anchors;
    anchors.reserve(2016);

    for (int l = 0; l < num_layers; ++l) {
        int s   = strides[l];
        int fms = input_size / s; // feature map size (square)
        for (int y = 0; y < fms; ++y) {
            for (int x = 0; x < fms; ++x) {
                float cx = (x + 0.5f) / fms;
                float cy = (y + 0.5f) / fms;
                for (int a = 0; a < anchors_per_cell; ++a) {
                    anchors.push_back({cx, cy});
                }
            }
        }
    }
    return anchors;
}

// ---- Box decoding -----------------------------------------------------------
// MediaPipe TensorsToDetectionsCalculator for palm_detection:
//   x_scale=y_scale=h_scale=w_scale=192, sigmoid_score=true
//   reverse_output_order=true  ← KEY: the model emits YXHW not XYWH
//   box format: [cy_enc, cx_enc, h_enc, w_enc, kp0_y, kp0_x, ..., kp6_y, kp6_x]
//
// Getting the order wrong here puts every detection at swapped coordinates
// and rotates the ROI 90° (because kp[0]→kp[2] direction is rotated), so the
// landmark crop is taken from a wrong region — the rest of the pipeline then
// produces landmarks at random locations relative to the actual hand.

static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static float iou(const PalmDetection& a, const PalmDetection& b) {
    float ax1 = a.cx - a.w * 0.5f, ax2 = a.cx + a.w * 0.5f;
    float ay1 = a.cy - a.h * 0.5f, ay2 = a.cy + a.h * 0.5f;
    float bx1 = b.cx - b.w * 0.5f, bx2 = b.cx + b.w * 0.5f;
    float by1 = b.cy - b.h * 0.5f, by2 = b.cy + b.h * 0.5f;
    float ix1 = std::max(ax1, bx1), ix2 = std::min(ax2, bx2);
    float iy1 = std::max(ay1, by1), iy2 = std::min(ay2, by2);
    if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;
    float inter = (ix2 - ix1) * (iy2 - iy1);
    float ua    = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}

std::vector<PalmDetection> decodePalmDetections(
    const float* scores,
    const float* boxes,
    const std::vector<Anchor>& anchors,
    float score_thresh,
    float nms_iou_thresh)
{
    constexpr float scale = 192.0f;
    constexpr int   num_kp = 7;
    const int n = static_cast<int>(anchors.size());

    // Decode all above-threshold detections
    std::vector<PalmDetection> dets;
    for (int i = 0; i < n; ++i) {
        float s = sigmoid(scores[i]);
        if (s < score_thresh) continue;

        const float* b  = boxes + i * 18;
        const Anchor& a = anchors[i];

        PalmDetection d{};
        d.score = s;
        // YXHW order — see comment block above.
        d.cy = a.cy + b[0] / scale;
        d.cx = a.cx + b[1] / scale;
        d.h  = std::exp(b[2] / scale);
        d.w  = std::exp(b[3] / scale);
        for (int k = 0; k < num_kp; ++k) {
            // Keypoint pairs are (y, x), not (x, y).
            d.kp_y[k] = a.cy + b[4 + 2 * k]     / scale;
            d.kp_x[k] = a.cx + b[4 + 2 * k + 1] / scale;
        }
        dets.push_back(d);
    }

    if (dets.empty()) return {};

    // NMS (weighted by score)
    std::sort(dets.begin(), dets.end(),
              [](const PalmDetection& a, const PalmDetection& b){ return a.score > b.score; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<PalmDetection> result;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > nms_iou_thresh)
                suppressed[j] = true;
        }
    }
    return result;
}
