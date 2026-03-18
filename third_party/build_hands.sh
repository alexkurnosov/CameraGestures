#!/bin/bash

# Build hand landmark tracking CPU library
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    //mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu \
    //mediapipe/framework:calculator_framework \
    //mediapipe/calculators/core:pass_through_calculator \
    //mediapipe/calculators/image:image_properties_calculator

# Create lib directory
mkdir -p lib/macos

# Copy built libraries
find bazel-bin -name "*.a" -o -name "*.dylib" | while read lib; do
    cp "$lib" lib/macos/
done

# Copy model files
mkdir -p models
cp mediapipe/modules/hand_landmark/*.tflite models/
cp mediapipe/modules/palm_detection/*.tflite models/

echo "MediaPipe build complete!"
