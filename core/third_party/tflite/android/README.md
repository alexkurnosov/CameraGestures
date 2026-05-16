# TFLite C++ prebuilts — Android

Version: **2.17.0**

Download the Android AAR and extract the `jni/` native libraries:

```
https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite/2.17.0/tensorflow-lite-2.17.0.aar
```

Extract `.so` files per ABI into subdirectories:

```
android/
  arm64-v8a/libtensorflowlite_jni.so
  armeabi-v7a/libtensorflowlite_jni.so
  x86_64/libtensorflowlite_jni.so
```

Headers are shared with `../include/`.

Run `core/scripts/vendor-tflite-android.sh` to automate this (Stage 3).
