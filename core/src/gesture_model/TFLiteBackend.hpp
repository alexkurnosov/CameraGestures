#pragma once
#include <string>
#include <vector>

/* Thin wrapper around the TFLite C API for single-input / single-output models.
 * Compiles to a no-op stub when CG_ENABLE_TFLITE is not defined. */

struct TfLiteModel;
struct TfLiteInterpreter;
struct TfLiteInterpreterOptions;

class TFLiteBackend {
public:
    TFLiteBackend();
    ~TFLiteBackend();

    TFLiteBackend(const TFLiteBackend&) = delete;
    TFLiteBackend& operator=(const TFLiteBackend&) = delete;

    bool load(const std::string& model_path);
    bool isLoaded() const { return interpreter_ != nullptr; }

    /* Size of input[0] and output[0] tensors (number of float elements). */
    int inputDim()  const;
    int outputDim() const;

    /* Run inference. input.size() must equal inputDim(). Returns false on error. */
    bool run(const std::vector<float>& input, std::vector<float>& output);

private:
    TfLiteModel*              model_       = nullptr;
    TfLiteInterpreterOptions* options_     = nullptr;
    TfLiteInterpreter*        interpreter_ = nullptr;
};
