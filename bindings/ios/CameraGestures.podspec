Pod::Spec.new do |s|
  s.name             = 'CameraGestures'
  s.version          = '0.5.0'   # Stage 5: HandGestureTypes + HandsRecognizing + GestureModel + HandGestureRecognizing
  s.summary          = 'Cross-platform gesture-recognition library — iOS binding'
  s.homepage         = 'https://github.com/yourname/CameraGestures'
  s.license          = { :type => 'MIT', :text => 'Private module — not for distribution' }
  s.author           = { 'Developer' => 'developer@example.com' }
  s.source           = { :path => '.' }
  s.platform         = :ios, '16.0'
  s.swift_version    = '5.0'

  # Prebuilt XCFrameworks produced by core/scripts/build-ios.sh.
  # CameraGestures.xcframework: C ABI static library + headers
  #   (CameraGestures.h, Types.h, HandsRecognizing.h, GestureModel.h,
  #    HandGestureRecognizing.h, module.modulemap → imported as CameraGesturesC).
  # TensorFlowLiteC.xcframework: vendored TFLite runtime (statically linked,
  #   no TFLite Swift pod needed).
  s.vendored_frameworks = [
    'XCFramework/CameraGestures.xcframework',
    '../../core/third_party/tflite/ios/TensorFlowLiteC.xcframework'
  ]

  # Swift wrapper sources.
  s.source_files = 'CameraGestures/**/*.swift'

  # hand_landmarker.task must be bundled so HandsRecognizingConfig can load it at runtime.
  # resource_bundles is used instead of resources so CocoaPods reliably creates a
  # CameraGesturesAssets.bundle build phase regardless of linkage mode.
  s.resource_bundles = {
    'CameraGesturesAssets' => ['hand_landmarker.task']
  }

  # Stage 3+: HandsRecognizing uses MediaPipeTasksVision for iOS landmark detection.
  s.dependency 'MediaPipeTasksVision', '0.10.14'
end
