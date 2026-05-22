#pragma once
#include <vector>
#include <cstdint>

struct Anchor { float cx, cy; };

// Box decoded from palm detection regression output.
// All coordinates normalized to [0,1] relative to the 192×192 input.
struct PalmDetection {
    float cx, cy;    // bounding box center
    float w, h;      // bounding box size
    float score;     // sigmoid confidence
    // 7 keypoints (wrist=0, middle_mcp=2 used for rotation)
    float kp_x[7];
    float kp_y[7];
};

// Generates the 2016 SSD anchors for the 192×192 palm detector.
// Mirrors MediaPipe SsdAnchorsCalculator with:
//   strides=[8,16,16,16], aspect_ratios=[1.0], fixed_anchor_size=true,
//   interpolated_scale_aspect_ratio=1.0 → 2 anchors per cell.
std::vector<Anchor> generatePalmAnchors();

// Decode raw TFLite outputs into filtered + NMS'd palm detections.
// scores: [num_anchors] raw logits
// boxes:  [num_anchors × 18] raw regressions (cx,cy,w,h + 7 kp pairs)
// anchors: from generatePalmAnchors()
std::vector<PalmDetection> decodePalmDetections(
    const float* scores,
    const float* boxes,
    const std::vector<Anchor>& anchors,
    float score_thresh = 0.5f,
    float nms_iou_thresh = 0.3f);
