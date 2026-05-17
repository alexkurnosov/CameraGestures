Pod::Spec.new do |s|
  s.name             = 'CameraGestures'
  s.version          = '0.1.0'   # Stage 3: HandGestureTypes + HandsRecognizing
  s.summary          = 'Cross-platform gesture-recognition library — iOS binding'
  s.homepage         = 'https://github.com/yourname/CameraGestures'
  s.license          = { :type => 'MIT', :text => 'Private module — not for distribution' }
  s.author           = { 'Developer' => 'developer@example.com' }
  s.source           = { :path => '.' }
  s.platform         = :ios, '16.0'
  s.swift_version    = '5.0'

  # Prebuilt XCFramework produced by core/scripts/build-ios.sh.
  # Contains the C ABI static library + headers (CameraGestures.h, Types.h,
  # HandsRecognizing.h, module.modulemap → imported as CameraGesturesC).
  s.vendored_frameworks = 'XCFramework/CameraGestures.xcframework'

  # Swift wrapper sources.
  s.source_files = 'CameraGestures/**/*.swift'

  # hand_landmarker.task must be bundled so HandsRecognizingConfig can load it at runtime.
  s.resources = '../../core/assets/hand_landmarker.task'

  # Stage 3+: HandsRecognizing uses MediaPipeTasksVision for iOS landmark detection.
  s.dependency 'MediaPipeTasksVision', '0.10.14'
end
