Pod::Spec.new do |s|
  s.name             = 'CameraGestures'
  s.version          = '0.0.1'   # Stage 2: HandGestureTypes only
  s.summary          = 'Cross-platform gesture-recognition library — iOS binding'
  s.homepage         = 'https://github.com/yourname/CameraGestures'
  s.license          = { :type => 'MIT', :text => 'Private module — not for distribution' }
  s.author           = { 'Developer' => 'developer@example.com' }
  s.source           = { :path => '.' }
  s.platform         = :ios, '16.0'
  s.swift_version    = '5.0'

  # Prebuilt XCFramework produced by core/scripts/build-ios.sh.
  # The XCFramework bundles the C ABI static library + headers
  # (CameraGestures.h, Types.h, module.modulemap → imported as CameraGesturesC).
  s.vendored_frameworks = 'XCFramework/CameraGestures.xcframework'

  # Swift wrapper sources — Stage 0.0.1: HandGestureTypes only.
  s.source_files = 'CameraGestures/**/*.swift'

  # Stage 3+: add MediaPipeTasksVision once HandsRecognizing lands.
  # s.dependency 'MediaPipeTasksVision', '~> 0.10.14'
end
