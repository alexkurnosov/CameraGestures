# TFLite C++ prebuilts — macOS

Version: **2.17.0**

Download the macOS C++ static library from the LiteRT GitHub releases:

```
https://github.com/google-ai-edge/LiteRT/releases/tag/v2.17.0
```

Expected artifact: `libtensorflowlite.a` (universal arm64 + x86_64).

Place it here as `macos/libtensorflowlite.a`.

Headers are shared with `../include/`.

Run `core/scripts/vendor-tflite-macos.sh` to automate this (Stage 4).
