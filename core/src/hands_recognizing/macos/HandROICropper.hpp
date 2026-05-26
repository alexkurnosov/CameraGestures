#pragma once
#include "AnchorDecoder.hpp"
#include <vector>
#include <cstdint>

// Rotated square ROI in normalized image coordinates [0,1].
struct HandROI {
    float cx, cy;     // center
    float half_side;  // half the side length of the square
    float angle;      // rotation angle in radians
};

// Compute hand ROI from a palm detection.
// Follows MediaPipe DetectionsToRectsCalculator + RectTransformationCalculator:
//   scale=2.6, shift_y=-0.5 (toward fingertips), square_long=true
//   rotation from kp[0] (wrist) to kp[2] (middle MCP)
HandROI palmDetectionToROI(const PalmDetection& det);

// Affine-warp the BGRA source image to a 224×224 RGB float32 buffer.
// Output: RGB interleaved, [224 × 224 × 3] floats, raw pixel values [0, 255].
// The hand_landmarks_detector.tflite model has baked-in ÷127.5−1 normalization,
// so callers must NOT pre-normalize — pass raw [0,255] float values.
void warpROI(
    const uint8_t* bgra_src,  // source BGRA8 frame
    int src_w, int src_h,     // source dimensions in pixels
    int src_stride,           // bytes per row in source
    const HandROI& roi,
    float* out_rgb224);       // pre-allocated [224*224*3] floats
