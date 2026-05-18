#include "TFLiteBackend.hpp"
#include <cstring>

#ifdef CG_ENABLE_TFLITE
#include "c_api.h"
#endif

/* -------------------------------------------------------------------------
 * Constructor / Destructor
 * ---------------------------------------------------------------------- */

TFLiteBackend::TFLiteBackend() = default;

TFLiteBackend::~TFLiteBackend() {
#ifdef CG_ENABLE_TFLITE
    if (interpreter_) TfLiteInterpreterDelete(interpreter_);
    if (options_)     TfLiteInterpreterOptionsDelete(options_);
    if (model_)       TfLiteModelDelete(model_);
#endif
}

/* -------------------------------------------------------------------------
 * load
 * ---------------------------------------------------------------------- */

bool TFLiteBackend::load(const std::string& model_path) {
#ifdef CG_ENABLE_TFLITE
    // Clean up any previous state
    if (interpreter_) { TfLiteInterpreterDelete(interpreter_); interpreter_ = nullptr; }
    if (options_)     { TfLiteInterpreterOptionsDelete(options_); options_ = nullptr; }
    if (model_)       { TfLiteModelDelete(model_); model_ = nullptr; }

    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_) return false;

    options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options_, 2);

    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    if (!interpreter_) return false;

    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) {
        TfLiteInterpreterDelete(interpreter_);
        interpreter_ = nullptr;
        return false;
    }
    return true;
#else
    (void)model_path;
    return false;
#endif
}

/* -------------------------------------------------------------------------
 * Dimension queries
 * ---------------------------------------------------------------------- */

int TFLiteBackend::inputDim() const {
#ifdef CG_ENABLE_TFLITE
    if (!interpreter_) return 0;
    const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    if (!t) return 0;
    int total = 1;
    for (int i = 0; i < TfLiteTensorNumDims(t); ++i)
        total *= TfLiteTensorDim(t, i);
    return total;
#else
    return 0;
#endif
}

int TFLiteBackend::outputDim() const {
#ifdef CG_ENABLE_TFLITE
    if (!interpreter_) return 0;
    const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    if (!t) return 0;
    int total = 1;
    for (int i = 0; i < TfLiteTensorNumDims(t); ++i)
        total *= TfLiteTensorDim(t, i);
    return total;
#else
    return 0;
#endif
}

/* -------------------------------------------------------------------------
 * run — copy input, invoke, copy output
 * ---------------------------------------------------------------------- */

bool TFLiteBackend::run(const std::vector<float>& input, std::vector<float>& output) {
#ifdef CG_ENABLE_TFLITE
    if (!interpreter_) return false;
    TfLiteTensor* in = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    if (!in) return false;

    if (TfLiteTensorCopyFromBuffer(in, input.data(),
                                   input.size() * sizeof(float)) != kTfLiteOk)
        return false;

    if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk)
        return false;

    const TfLiteTensor* out = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    if (!out) return false;

    int n = outputDim();
    output.resize(n);
    TfLiteTensorCopyToBuffer(out, output.data(), n * sizeof(float));
    return true;
#else
    (void)input; (void)output;
    return false;
#endif
}
