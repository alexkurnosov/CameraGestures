// macOS HandLandmarker — LiteRT-only, two-stage decode.
// SPIKE STATUS: IMPLEMENTED (Stage 3 step 6 / Stage 7).
//
// Pipeline:
//   1. Resize BGRA frame to 192×192 RGB[-1,1], run palm detector.
//   2. Decode SSD detections → NMS → ROI per hand.
//   3. Affine-warp ROI to 224×224 RGB[-1,1], run landmark model.
//   4. Project 21 landmarks back to original image coordinates.
//   5. Push cg_handshot into the attached HandsRecognizer.

#include "HandLandmarkerLiteRT.hpp"
#include "TaskBundleParser.hpp"
#include "AnchorDecoder.hpp"
#include "HandROICropper.hpp"
#include "CameraGestures/Types.h"

#ifdef CG_ENABLE_TFLITE
#include "c_api.h"
#endif

#include <cmath>
#include <algorithm>
#include <vector>

// ---- Constructor / Destructor -----------------------------------------------

HandLandmarkerLiteRT::HandLandmarkerLiteRT() {
    det_input_buf_.resize(1 * 192 * 192 * 3);
    lm_input_buf_.resize(1 * 224 * 224 * 3);
}

HandLandmarkerLiteRT::~HandLandmarkerLiteRT() {
#ifdef CG_ENABLE_TFLITE
    if (det_interp_)  TfLiteInterpreterDelete(det_interp_);
    if (det_options_) TfLiteInterpreterOptionsDelete(det_options_);
    if (det_model_)   TfLiteModelDelete(det_model_);
    if (lm_interp_)   TfLiteInterpreterDelete(lm_interp_);
    if (lm_options_)  TfLiteInterpreterOptionsDelete(lm_options_);
    if (lm_model_)    TfLiteModelDelete(lm_model_);
#endif
}

// ---- Load -------------------------------------------------------------------

bool HandLandmarkerLiteRT::load(const std::string& task_path) {
#ifndef CG_ENABLE_TFLITE
    (void)task_path;
    return false;
#else
    TaskBundleParser bundle(task_path);
    if (!bundle.isOpen()) return false;

    auto det_data = bundle.extract("hand_detector.tflite");
    auto lm_data  = bundle.extract("hand_landmarks_detector.tflite");
    if (det_data.empty() || lm_data.empty()) return false;

    if (!loadDetector(det_data) || !loadLandmarker(lm_data)) return false;

    anchors_ = generatePalmAnchors();
    loaded_  = true;
    return true;
#endif
}

bool HandLandmarkerLiteRT::loadDetector(const std::vector<uint8_t>& data) {
#ifndef CG_ENABLE_TFLITE
    return false;
#else
    det_model_ = TfLiteModelCreate(data.data(), data.size());
    if (!det_model_) return false;
    det_options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(det_options_, 2);
    det_interp_ = TfLiteInterpreterCreate(det_model_, det_options_);
    if (!det_interp_) return false;
    return TfLiteInterpreterAllocateTensors(det_interp_) == kTfLiteOk;
#endif
}

bool HandLandmarkerLiteRT::loadLandmarker(const std::vector<uint8_t>& data) {
#ifndef CG_ENABLE_TFLITE
    return false;
#else
    lm_model_ = TfLiteModelCreate(data.data(), data.size());
    if (!lm_model_) return false;
    lm_options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(lm_options_, 2);
    lm_interp_ = TfLiteInterpreterCreate(lm_model_, lm_options_);
    if (!lm_interp_) return false;
    return TfLiteInterpreterAllocateTensors(lm_interp_) == kTfLiteOk;
#endif
}

// ---- Internal helpers -------------------------------------------------------

#ifdef CG_ENABLE_TFLITE
// Find the output tensor with exactly `target_size` total elements.
static const TfLiteTensor* findOutputBySize(TfLiteInterpreter* interp, int target_size) {
    int n = TfLiteInterpreterGetOutputTensorCount(interp);
    for (int i = 0; i < n; ++i) {
        const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interp, i);
        int sz = 1;
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
            sz *= TfLiteTensorDim(t, d);
        if (sz == target_size) return t;
    }
    return nullptr;
}

// Find the first or second output tensor with exactly 1 total element.
static const TfLiteTensor* findOutputBySize1(TfLiteInterpreter* interp, int occurrence) {
    int n = TfLiteInterpreterGetOutputTensorCount(interp);
    int found = 0;
    for (int i = 0; i < n; ++i) {
        const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interp, i);
        int sz = 1;
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
            sz *= TfLiteTensorDim(t, d);
        if (sz == 1) {
            if (found == occurrence) return t;
            ++found;
        }
    }
    return nullptr;
}
#endif

// ---- Main frame processing --------------------------------------------------

void HandLandmarkerLiteRT::pushFrame(
    const uint8_t* bgra, int width, int height, int stride, double timestamp)
{
    auto emit_absent = [&]() {
        cg_handshot absent{};
        absent.is_absent  = 1;
        absent.timestamp  = timestamp;
        absent.handedness = CG_HAND_UNKNOWN;
        cg_hands_recognizer_push_handshot(recognizer_, &absent);
    };

    if (!loaded_ || !recognizer_) { emit_absent(); return; }

#ifndef CG_ENABLE_TFLITE
    emit_absent();
#else
    // ---- Step 1: Resize to 192×192 and normalize to RGB [-1,1] -----------
    constexpr int det_size = 192;
    float* det_inp = det_input_buf_.data();
    for (int y = 0; y < det_size; ++y) {
        for (int x = 0; x < det_size; ++x) {
            int sx = std::max(0, std::min(width  - 1, (int)((x + 0.5f) / det_size * width)));
            int sy = std::max(0, std::min(height - 1, (int)((y + 0.5f) / det_size * height)));
            const uint8_t* px = bgra + sy * stride + sx * 4;
            int idx = (y * det_size + x) * 3;
            det_inp[idx + 0] = px[2] / 127.5f - 1.0f; // R
            det_inp[idx + 1] = px[1] / 127.5f - 1.0f; // G
            det_inp[idx + 2] = px[0] / 127.5f - 1.0f; // B
        }
    }

    // ---- Run palm detector ------------------------------------------------
    TfLiteTensor* det_in = TfLiteInterpreterGetInputTensor(det_interp_, 0);
    if (!det_in) { emit_absent(); return; }
    TfLiteTensorCopyFromBuffer(det_in, det_inp, det_input_buf_.size() * sizeof(float));
    if (TfLiteInterpreterInvoke(det_interp_) != kTfLiteOk) { emit_absent(); return; }

    // Identify output tensors by size: scores=[2016], boxes=[2016*18]
    const TfLiteTensor* scores_t = findOutputBySize(det_interp_, 2016);
    const TfLiteTensor* boxes_t  = findOutputBySize(det_interp_, 2016 * 18);
    if (!scores_t || !boxes_t) { emit_absent(); return; }

    std::vector<float> scores_data(2016), boxes_data(2016 * 18);
    TfLiteTensorCopyToBuffer(scores_t, scores_data.data(), scores_data.size() * sizeof(float));
    TfLiteTensorCopyToBuffer(boxes_t,  boxes_data.data(),  boxes_data.size()  * sizeof(float));

    auto detections = decodePalmDetections(scores_data.data(), boxes_data.data(), anchors_);
    if (detections.empty()) { emit_absent(); return; }

    // ---- Step 2: Affine-warp detected palm to 224×224 --------------------
    const PalmDetection& det = detections[0];
    HandROI roi = palmDetectionToROI(det);
    warpROI(bgra, width, height, stride, roi, lm_input_buf_.data());

    // ---- Step 3: Run landmark model --------------------------------------
    TfLiteTensor* lm_in = TfLiteInterpreterGetInputTensor(lm_interp_, 0);
    if (!lm_in) { emit_absent(); return; }
    TfLiteTensorCopyFromBuffer(lm_in, lm_input_buf_.data(),
                               lm_input_buf_.size() * sizeof(float));
    if (TfLiteInterpreterInvoke(lm_interp_) != kTfLiteOk) { emit_absent(); return; }

    // Landmark output: [21*3=63 floats]; presence and handedness are scalars.
    const TfLiteTensor* lm_t       = findOutputBySize(lm_interp_, 63);
    const TfLiteTensor* presence_t = findOutputBySize1(lm_interp_, 0);
    const TfLiteTensor* handed_t   = findOutputBySize1(lm_interp_, 1);
    if (!lm_t) { emit_absent(); return; }

    float presence = 0.5f;
    if (presence_t) TfLiteTensorCopyToBuffer(presence_t, &presence, sizeof(float));
    // sigmoid for presence score
    float pres_sig = 1.0f / (1.0f + std::exp(-presence));
    if (pres_sig < 0.5f) { emit_absent(); return; }

    float handedness_raw = 0.0f;
    if (handed_t) TfLiteTensorCopyToBuffer(handed_t, &handedness_raw, sizeof(float));

    float lm_data[63];
    TfLiteTensorCopyToBuffer(lm_t, lm_data, sizeof(lm_data));

    // ---- Step 4: Project landmarks back to original image ----------------
    cg_handshot shot{};
    shot.timestamp  = timestamp;
    shot.is_absent  = 0;
    // handedness: model outputs ~0 for left, ~1 for right (after sigmoid)
    float h_sig = 1.0f / (1.0f + std::exp(-handedness_raw));
    shot.handedness = h_sig > 0.5f ? CG_HAND_RIGHT : CG_HAND_LEFT;

    const float cos_a = std::cos(roi.angle);
    const float sin_a = std::sin(roi.angle);
    const float side  = roi.half_side * 2.0f;

    for (int k = 0; k < 21; ++k) {
        float lx = lm_data[k * 3 + 0]; // normalized [0,1] in crop space
        float ly = lm_data[k * 3 + 1];
        float lz = lm_data[k * 3 + 2]; // depth (scale TBD)

        // Map from crop [0,1] to ROI-frame relative offset
        float ds = (lx - 0.5f) * side;
        float dt = (ly - 0.5f) * side;

        // Rotate to image normalized coords [0,1]
        float img_x = roi.cx + ds * cos_a - dt * sin_a;
        float img_y = roi.cy + ds * sin_a + dt * cos_a;

        shot.landmarks[k] = cg_point3d{img_x, img_y, lz};
    }

    cg_hands_recognizer_push_handshot(recognizer_, &shot);
#endif
}

// ---- C ABI ------------------------------------------------------------------

struct cg_hand_landmarker_lrt_s { HandLandmarkerLiteRT impl; };

extern "C" {

cg_hand_landmarker_lrt_ref cg_hand_landmarker_lrt_create(
    const char* task_path,
    cg_hands_recognizer_ref recognizer)
{
    auto* obj = new cg_hand_landmarker_lrt_s{};
    obj->impl.setRecognizer(recognizer);
    if (!obj->impl.load(task_path)) { delete obj; return nullptr; }
    return obj;
}

void cg_hand_landmarker_lrt_destroy(cg_hand_landmarker_lrt_ref ref) {
    delete ref;
}

void cg_hand_landmarker_lrt_push_frame(
    cg_hand_landmarker_lrt_ref ref,
    const uint8_t* bgra,
    int width, int height, int stride,
    double timestamp)
{
    if (ref) ref->impl.pushFrame(bgra, width, height, stride, timestamp);
}

} // extern "C"
