#pragma once
#include "CameraGestures/HandsRecognizing.h"
#include <string>
#include <vector>
#include <memory>

// Forward-declare TFLite types to avoid leaking headers into consumers.
struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;

struct Anchor;

// LiteRT-only macOS HandLandmarker.
// Loads hand_landmarker.task, runs two-stage inference (palm detector →
// landmark model), and pushes cg_handshot results into the supplied recognizer.
//
// Thread-safety: not thread-safe; call pushFrame from a single thread.
class HandLandmarkerLiteRT {
public:
    HandLandmarkerLiteRT();
    ~HandLandmarkerLiteRT();

    // Load the .task bundle and prepare both TFLite interpreters.
    // Returns true on success.
    bool load(const std::string& task_path);

    // Associate the HandsRecognizer that receives handshot callbacks.
    void setRecognizer(cg_hands_recognizer_ref ref) { recognizer_ = ref; }

    // Push a BGRA8 frame. On success, fires cg_hands_recognizer_push_handshot
    // on the attached recognizer (one shot per detected hand, plus absent shots).
    void pushFrame(const uint8_t* bgra, int width, int height, int stride, double timestamp);

    bool isLoaded() const { return loaded_; }

private:
    bool loadDetector(const std::vector<uint8_t>& model_data);
    bool loadLandmarker(const std::vector<uint8_t>& model_data);
    bool runDetector(const uint8_t* bgra, int w, int h, int stride);
    bool runLandmarker(const uint8_t* bgra, int w, int h, int stride,
                       float det_cx, float det_cy, float det_half_side, float det_angle,
                       double timestamp);

    // TFLite handles for palm detector
    TfLiteModel*              det_model_       = nullptr;
    TfLiteInterpreterOptions* det_options_     = nullptr;
    TfLiteInterpreter*        det_interp_      = nullptr;

    // TFLite handles for landmark model
    TfLiteModel*              lm_model_        = nullptr;
    TfLiteInterpreterOptions* lm_options_      = nullptr;
    TfLiteInterpreter*        lm_interp_       = nullptr;

    // Pre-computed SSD anchors for the palm detector
    std::vector<Anchor>       anchors_;

    // Attached recognizer
    cg_hands_recognizer_ref   recognizer_  = nullptr;

    bool                      loaded_      = false;

    // Scratch buffers (avoid heap alloc per frame)
    std::vector<float>  det_input_buf_;   // [1 * 192 * 192 * 3]
    std::vector<float>  lm_input_buf_;    // [1 * 224 * 224 * 3]
};

// ---- C ABI (macOS only) -----------------------------------------------------
// Exposed so the macOS Swift binding can drive inference without Obj-C++.

extern "C" {

typedef struct cg_hand_landmarker_lrt_s* cg_hand_landmarker_lrt_ref;

// Create a LiteRT HandLandmarker and attach it to an existing recognizer.
// task_path: absolute path to hand_landmarker.task.
cg_hand_landmarker_lrt_ref cg_hand_landmarker_lrt_create(
    const char*             task_path,
    cg_hands_recognizer_ref recognizer);

void cg_hand_landmarker_lrt_destroy(cg_hand_landmarker_lrt_ref ref);

// Push a BGRA8 frame for inference (timestamp in seconds since epoch).
void cg_hand_landmarker_lrt_push_frame(
    cg_hand_landmarker_lrt_ref ref,
    const uint8_t* bgra,
    int width, int height, int stride,
    double timestamp);

} // extern "C"
