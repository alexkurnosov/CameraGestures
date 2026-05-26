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
#include "xnnpack_delegate.h"
#include <cstdarg>
#include <cstdio>
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

// TFLite error reporter — forwards to stderr so Xcode console shows errors.
#ifdef CG_ENABLE_TFLITE
static void tfliteErrorReporter(void* /*user*/, const char* fmt, va_list args) {
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, args);
    fprintf(stderr, "[TFLite] %s\n", buf);
}

// Create interpreter options with:
//  - error reporter for diagnostics
//  - XNNPACK delegate explicitly set to single-threaded mode
//    (avoids re-platformed iOS-Sim XNNPACK thread-pool crash on macOS)
static TfLiteInterpreterOptions* makeSafeOptions() {
    TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetErrorReporter(opts, tfliteErrorReporter, nullptr);
    TfLiteInterpreterOptionsSetNumThreads(opts, 1);

    // Explicitly supply an XNNPACK delegate with num_threads=0 (no thread pool).
    // This overrides the auto-applied XNNPACK delegate whose thread-pool
    // initialisation crashes when the re-platformed iOS-Sim binary runs on macOS.
    TfLiteXNNPackDelegateOptions xnn = TfLiteXNNPackDelegateOptionsDefault();
    xnn.num_threads = 0;  // single-threaded; avoids pthreads pool on macOS
    TfLiteDelegate* xnn_del = TfLiteXNNPackDelegateCreate(&xnn);
    if (xnn_del) {
        TfLiteInterpreterOptionsAddDelegate(opts,
            reinterpret_cast<TfLiteOpaqueDelegate*>(xnn_del));
    }
    return opts;
}
#endif

bool HandLandmarkerLiteRT::loadDetector(const std::vector<uint8_t>& data) {
#ifndef CG_ENABLE_TFLITE
    return false;
#else
    det_model_ = TfLiteModelCreate(data.data(), data.size());
    if (!det_model_) return false;
    det_options_ = makeSafeOptions();
    det_interp_ = TfLiteInterpreterCreate(det_model_, det_options_);
    if (!det_interp_) return false;
    if (TfLiteInterpreterAllocateTensors(det_interp_) != kTfLiteOk) {
        fprintf(stderr, "[TFLite] AllocateTensors failed for palm detector\n");
        return false;
    }
    // Dump input tensor shape
    {
        const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(det_interp_, 0);
        fprintf(stderr, "[CG:Debug] palm detector input[0] name=%s dims=[",
                t ? TfLiteTensorName(t) : "?");
        if (t) {
            for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
                fprintf(stderr, "%d%s", TfLiteTensorDim(t, d),
                        d < TfLiteTensorNumDims(t)-1 ? "," : "");
        }
        fprintf(stderr, "]\n");
    }
    // Dump output tensor shapes and types
    {
        static const char* type_str[] = {
            "NoType","Float32","Int32","UInt8","Int64","String","Bool",
            "Int16","Complex64","Int8","Float16","Float64"};
        auto tname = [&](TfLiteType t) -> const char* {
            int v = static_cast<int>(t);
            return (v >= 0 && v < 12) ? type_str[v] : "?";
        };
        int nOut = TfLiteInterpreterGetOutputTensorCount(det_interp_);
        fprintf(stderr, "[CG:Debug] palm detector: %d output tensors:\n", nOut);
        for (int i = 0; i < nOut; ++i) {
            const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(det_interp_, i);
            int sz = 1;
            fprintf(stderr, "[CG:Debug]   out[%d] name=%-30s type=%-8s bytes=%-8zu dims=[", i,
                    t ? TfLiteTensorName(t) : "?",
                    t ? tname(TfLiteTensorType(t)) : "?",
                    t ? TfLiteTensorByteSize(t) : 0);
            if (t) {
                for (int d = 0; d < TfLiteTensorNumDims(t); ++d) {
                    sz *= TfLiteTensorDim(t, d);
                    fprintf(stderr, "%d%s", TfLiteTensorDim(t, d),
                            d < TfLiteTensorNumDims(t)-1 ? "," : "");
                }
            }
            fprintf(stderr, "] total=%d\n", sz);
        }
    }
    return true;
#endif
}

bool HandLandmarkerLiteRT::loadLandmarker(const std::vector<uint8_t>& data) {
#ifndef CG_ENABLE_TFLITE
    return false;
#else
    lm_model_ = TfLiteModelCreate(data.data(), data.size());
    if (!lm_model_) return false;
    lm_options_ = makeSafeOptions();
    lm_interp_ = TfLiteInterpreterCreate(lm_model_, lm_options_);
    if (!lm_interp_) return false;
    if (TfLiteInterpreterAllocateTensors(lm_interp_) != kTfLiteOk) {
        fprintf(stderr, "[TFLite] AllocateTensors failed for landmark model\n");
        return false;
    }
    // Dump input tensor shape
    {
        const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(lm_interp_, 0);
        fprintf(stderr, "[CG:Debug] landmark model input[0] name=%s dims=[",
                t ? TfLiteTensorName(t) : "?");
        if (t) {
            for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
                fprintf(stderr, "%d%s", TfLiteTensorDim(t, d),
                        d < TfLiteTensorNumDims(t)-1 ? "," : "");
        }
        fprintf(stderr, "]\n");
    }
    // Dump output tensor shapes and types
    {
        static const char* type_str[] = {
            "NoType","Float32","Int32","UInt8","Int64","String","Bool",
            "Int16","Complex64","Int8","Float16","Float64"};
        auto tname = [&](TfLiteType t) -> const char* {
            int v = static_cast<int>(t);
            return (v >= 0 && v < 12) ? type_str[v] : "?";
        };
        int nOut = TfLiteInterpreterGetOutputTensorCount(lm_interp_);
        fprintf(stderr, "[CG:Debug] landmark model: %d output tensors:\n", nOut);
        for (int i = 0; i < nOut; ++i) {
            const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(lm_interp_, i);
            int sz = 1;
            fprintf(stderr, "[CG:Debug]   out[%d] name=%-30s type=%-8s bytes=%-8zu dims=[", i,
                    t ? TfLiteTensorName(t) : "?",
                    t ? tname(TfLiteTensorType(t)) : "?",
                    t ? TfLiteTensorByteSize(t) : 0);
            if (t) {
                for (int d = 0; d < TfLiteTensorNumDims(t); ++d) {
                    sz *= TfLiteTensorDim(t, d);
                    fprintf(stderr, "%d%s", TfLiteTensorDim(t, d),
                            d < TfLiteTensorNumDims(t)-1 ? "," : "");
                }
            }
            fprintf(stderr, "] total=%d\n", sz);
        }
    }
    return true;
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
    // Debug frame counter — print a heartbeat every 90 frames (~3 s at 30 fps).
    static int dbg_frame = 0;
    ++dbg_frame;
    const bool dbg_log = (dbg_frame % 90 == 0);
    if (dbg_log)
        fprintf(stderr, "[CG:LiteRT] frame %d  %dx%d  loaded=%d  recognizer=%d\n",
                dbg_frame, width, height, loaded_ ? 1 : 0, recognizer_ ? 1 : 0);

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
    // ---- Step 1: Resize to 192×192, normalize to RGB [-1,1] ---------------
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

    // 0.6 sigmoid ≈ logit > 0.4 — enough to suppress background noise while
    // still catching real hands.  The 5-frame gate guard is the main FP shield.
    auto detections = decodePalmDetections(scores_data.data(), boxes_data.data(), anchors_, 0.6f);

    // Log the best raw detection score every 90 frames so we can see if the detector
    // is producing any signal at all (even below the 0.7 threshold).
    if (dbg_log) {
        float best_sig = 0.0f;
        float best_raw = -999.0f;
        for (int i = 0; i < 2016; ++i) {
            float s = 1.0f / (1.0f + std::exp(-scores_data[i]));
            if (s > best_sig) { best_sig = s; best_raw = scores_data[i]; }
        }
        // Sample first few raw score values and first few raw box values
        fprintf(stderr, "[CG:LiteRT] palm best_score_sig=%.3f  best_raw=%.4f  detections=%d\n",
                best_sig, best_raw, (int)detections.size());
        fprintf(stderr, "[CG:LiteRT] scores[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
                scores_data[0], scores_data[1], scores_data[2], scores_data[3], scores_data[4]);
        fprintf(stderr, "[CG:LiteRT] boxes[0..5]:  %.4f %.4f %.4f %.4f %.4f %.4f\n",
                boxes_data[0], boxes_data[1], boxes_data[2], boxes_data[3],
                boxes_data[4], boxes_data[5]);
    }

    if (detections.empty()) { emit_absent(); return; }

    // Per-frame diagnostic: where does the detector think the palms are?
    // Coords are in normalized [0,1] of the (squashed) 192×192 model input.
    // (0.0, 0.0) = top-left, (1.0, 1.0) = bottom-right of the resized image.
    if (dbg_log) {
        const int nlog = std::min((int)detections.size(), 5);
        for (int i = 0; i < nlog; ++i) {
            const auto& d = detections[i];
            fprintf(stderr,
                "[CG:Det] top[%d] score=%.3f  box=(cx=%.2f cy=%.2f w=%.2f h=%.2f)"
                "  wrist=(%.2f,%.2f) mcp2=(%.2f,%.2f)\n",
                i, d.score, d.cx, d.cy, d.w, d.h,
                d.kp_x[0], d.kp_y[0], d.kp_x[2], d.kp_y[2]);
        }
    }

    // ---- Steps 2-4: Try top-N detections; pick the one with highest presence --
    // The palm detector fires ~25 false-positive detections per frame on typical
    // indoor scenes.  We run the landmark model on up to kMaxTry candidates and
    // keep the one with the highest presence score (out[2]).  When a real hand
    // is present it should score slightly higher than the false-positive crops.
    // Limit to kMaxTry to keep per-frame inference cost bounded.
    constexpr int kMaxTry = 5;
    const int n_try = std::min((int)detections.size(), kMaxTry);

    // Pre-fetch landmark interpreter handles — they don't change between runs.
    TfLiteTensor* lm_in = TfLiteInterpreterGetInputTensor(lm_interp_, 0);
    if (!lm_in) { emit_absent(); return; }
    // Find output tensors once (same graph layout every Invoke).
    const TfLiteTensor* lm_t       = nullptr;
    const TfLiteTensor* presence_t = nullptr;
    const TfLiteTensor* handed_t   = nullptr;

    // Try all n_try candidates; keep the one with the highest presence score.
    // out[2] (Identity_2) is our best proxy for presence — it varies ~0.60-0.69,
    // not dramatically, but picking the MAX reliably favours the real hand crop
    // over false-positive crops when both compete in the same frame.
    // Accept anything above 0.5 (the score doesn't discriminate below that).
    constexpr float kPresenceThresh = 0.5f;

    bool  found_hand          = false;
    float best_pres_sig       = 0.0f;
    HandROI found_roi{};
    float   found_handedness_raw = 0.0f;
    float   found_lm_data[63]{};
    float   candidate_lm[63]{};
    HandROI candidate_roi{};
    float   candidate_handed = 0.0f;

    for (int di = 0; di < n_try; ++di) {
        const PalmDetection& det = detections[di];

        // Step 2: affine-warp ROI to 224×224
        HandROI roi = palmDetectionToROI(det);
        warpROI(bgra, width, height, stride, roi, lm_input_buf_.data());

        // Step 3: run landmark model
        TfLiteTensorCopyFromBuffer(lm_in, lm_input_buf_.data(),
                                   lm_input_buf_.size() * sizeof(float));
        if (TfLiteInterpreterInvoke(lm_interp_) != kTfLiteOk) continue;

        // Cache output tensor pointers (same graph layout every Invoke).
        // Model outputs: out[0]=screen_lm[63], out[1]=handedness, out[2]=presence,
        //                out[3]=world_lm[63].
        if (!lm_t) {
            lm_t       = findOutputBySize(lm_interp_, 63);  // out[0]: screen landmarks
            presence_t = findOutputBySize1(lm_interp_, 1);  // out[2]: presence proxy
            handed_t   = findOutputBySize1(lm_interp_, 0);  // out[1]: handedness
        }
        if (!lm_t) continue;

        float scalar_h = 0.0f, scalar_p = 0.0f;
        if (handed_t)   TfLiteTensorCopyToBuffer(handed_t,   &scalar_h, sizeof(float));
        if (presence_t) TfLiteTensorCopyToBuffer(presence_t, &scalar_p, sizeof(float));
        float sig_h = 1.0f / (1.0f + std::exp(-scalar_h));
        float sig_p = 1.0f / (1.0f + std::exp(-scalar_p));

        fprintf(stderr, "[CG:LiteRT] det[%d/%d] palm=%.3f  h=%.3f  pres=%.3f\n",
                di, n_try, det.score, sig_h, sig_p);

        if (sig_p < kPresenceThresh) continue;

        // ---- Hand-shape validity check ------------------------------------
        // The landmark model outputs x,y in PIXEL coords of the 224×224 input
        // crop (range [0, 224]), so we normalize by the input size first.
        // For non-hand crops the model still outputs landmarks that look
        // hand-like (it always emits 21 points), so this filter only catches
        // egregious failures — its main job is sanity, not full discrimination.
        constexpr float kLmInputSize = 224.0f;
        float lm_tmp[63];
        TfLiteTensorCopyToBuffer(lm_t, lm_tmp, sizeof(lm_tmp));

        // Normalize x and y to [0,1] crop space (keep z untouched — it's depth).
        for (int i = 0; i < 21; ++i) {
            lm_tmp[i * 3 + 0] /= kLmInputSize;
            lm_tmp[i * 3 + 1] /= kLmInputSize;
        }

        const float wrist_y   = lm_tmp[0 * 3 + 1];
        const float mid_tip_y = lm_tmp[12 * 3 + 1];
        float min_x = 1.0f, max_x = 0.0f, min_y = 1.0f, max_y = 0.0f;
        for (int i = 0; i < 21; ++i) {
            float x = lm_tmp[i * 3 + 0];
            float y = lm_tmp[i * 3 + 1];
            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
        }
        const float spread_x = max_x - min_x;
        const float spread_y = max_y - min_y;

        // Orientation: wrist must be below center, middle tip must be above.
        // Spread: landmarks must cover ≥30% of the crop in at least one axis.
        const bool orient_ok = (wrist_y > 0.50f && mid_tip_y < 0.50f);
        const bool spread_ok = (spread_x > 0.30f || spread_y > 0.30f);
        if (!orient_ok || !spread_ok) {
            fprintf(stderr, "[CG:LiteRT]   reject: wrist_y=%.2f mid_y=%.2f sx=%.2f sy=%.2f\n",
                    wrist_y, mid_tip_y, spread_x, spread_y);
            continue;
        }

        // Keep the candidate with the highest presence score.
        if (!found_hand || sig_p > best_pres_sig) {
            best_pres_sig    = sig_p;
            found_hand       = true;
            candidate_roi    = roi;
            candidate_handed = scalar_h;
            std::copy(lm_tmp, lm_tmp + 63, candidate_lm);
        }
    }

    if (!found_hand) { emit_absent(); return; }

    found_roi             = candidate_roi;
    found_handedness_raw  = candidate_handed;
    std::copy(candidate_lm, candidate_lm + 63, found_lm_data);

    // ---- Step 4: Project landmarks back to original image ----------------
    cg_handshot shot{};
    shot.timestamp  = timestamp;
    shot.is_absent  = 0;
    // handedness: model outputs ~0 for left, ~1 for right (after sigmoid)
    float h_sig = 1.0f / (1.0f + std::exp(-found_handedness_raw));
    shot.handedness = h_sig > 0.5f ? CG_HAND_RIGHT : CG_HAND_LEFT;

    const float cos_a = std::cos(found_roi.angle);
    const float sin_a = std::sin(found_roi.angle);
    const float side  = found_roi.half_side * 2.0f;

    for (int k = 0; k < 21; ++k) {
        float lx = found_lm_data[k * 3 + 0]; // normalized [0,1] in crop space
        float ly = found_lm_data[k * 3 + 1];
        float lz = found_lm_data[k * 3 + 2]; // depth (scale TBD)

        // Map from crop [0,1] to ROI-frame relative offset
        float ds = (lx - 0.5f) * side;
        float dt = (ly - 0.5f) * side;

        // Rotate to image normalized coords [0,1]
        float img_x = found_roi.cx + ds * cos_a - dt * sin_a;
        float img_y = found_roi.cy + ds * sin_a + dt * cos_a;

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
