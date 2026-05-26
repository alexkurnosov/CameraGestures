#include "HandROICropper.hpp"
#include <cmath>
#include <algorithm>

// ---- ROI from palm detection ------------------------------------------------

HandROI palmDetectionToROI(const PalmDetection& det) {
    // Rotation: wrist (kp[0]) → middle MCP (kp[2])
    // MediaPipe target angle = 90° → subtract pi/2 from atan2
    float dx = det.kp_x[2] - det.kp_x[0];
    float dy = det.kp_y[2] - det.kp_y[0];
    float angle = std::atan2(dy, dx) - static_cast<float>(M_PI_2);

    // Bounding box center and long side
    float long_side = std::max(det.w, det.h);

    // RectTransformationCalculator: scale=2.6, shift_y=-0.5, square_long=true
    constexpr float scale   = 2.6f;
    constexpr float shift_y = -0.5f;

    float new_side = long_side * scale;  // square after scaling

    // Shift center toward fingertips (in the rotated frame, -y = up = fingertips)
    // new_cx = cx + new_w * (-sin(angle) * shift_y)
    // new_cy = cy + new_h * ( cos(angle) * shift_y)
    float cx = det.cx + new_side * (-std::sin(angle) * shift_y);
    float cy = det.cy + new_side * ( std::cos(angle) * shift_y);

    return HandROI{cx, cy, new_side * 0.5f, angle};
}

// ---- Affine warp with bilinear sampling -------------------------------------

static float bilinearSample(
    const uint8_t* bgra,
    int src_w, int src_h, int src_stride,
    float px, float py,
    int channel_bgra)  // B=0,G=1,R=2,A=3
{
    // Clamp to valid range
    px = std::max(0.0f, std::min(static_cast<float>(src_w  - 1), px));
    py = std::max(0.0f, std::min(static_cast<float>(src_h - 1), py));

    int x0 = static_cast<int>(px), y0 = static_cast<int>(py);
    int x1 = std::min(x0 + 1, src_w  - 1);
    int y1 = std::min(y0 + 1, src_h - 1);
    float fx = px - x0, fy = py - y0;

    auto px_val = [&](int x, int y) -> float {
        return static_cast<float>(bgra[y * src_stride + x * 4 + channel_bgra]);
    };

    float v00 = px_val(x0, y0), v10 = px_val(x1, y0);
    float v01 = px_val(x0, y1), v11 = px_val(x1, y1);
    return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
         + v01 * (1 - fx) *      fy  + v11 * fx *      fy;
}

void warpROI(
    const uint8_t* bgra_src,
    int src_w, int src_h, int src_stride,
    const HandROI& roi,
    float* out_rgb224)
{
    constexpr int out_size = 224;
    const float cos_a = std::cos(roi.angle);
    const float sin_a = std::sin(roi.angle);
    // full side in normalized coords
    const float side_n = roi.half_side * 2.0f;

    for (int v = 0; v < out_size; ++v) {
        for (int u = 0; u < out_size; ++u) {
            // Normalized position in output [0,1]
            float s = (u + 0.5f) / out_size;
            float t = (v + 0.5f) / out_size;

            // Map to ROI-frame offsets (relative to ROI center, in normalized image coords)
            float ds = (s - 0.5f) * side_n;
            float dt = (t - 0.5f) * side_n;

            // Rotate to image frame
            float img_x = roi.cx + ds * cos_a - dt * sin_a;
            float img_y = roi.cy + ds * sin_a + dt * cos_a;

            // Convert to pixel coords
            float px = img_x * src_w;
            float py = img_y * src_h;

            // Sample BGRA → convert to RGB.
            // Raw float [0,255]: the hand_landmarks_detector.tflite model has
            // its own normalization (÷127.5 − 1) baked in as initial ops, so
            // the caller must NOT pre-normalize — just pass raw pixel values.
            float b = bilinearSample(bgra_src, src_w, src_h, src_stride, px, py, 0);
            float g = bilinearSample(bgra_src, src_w, src_h, src_stride, px, py, 1);
            float r = bilinearSample(bgra_src, src_w, src_h, src_stride, px, py, 2);

            int idx = (v * out_size + u) * 3;
            out_rgb224[idx + 0] = r;  // R  [0,255]
            out_rgb224[idx + 1] = g;  // G  [0,255]
            out_rgb224[idx + 2] = b;  // B  [0,255]
        }
    }
}
