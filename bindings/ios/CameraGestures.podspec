Pod::Spec.new do |s|
  s.name             = 'CameraGestures'
  s.version          = '0.1.0'
  s.summary          = 'Cross-platform gesture-recognition library — iOS binding'
  s.homepage         = 'https://github.com/yourname/CameraGestures'
  s.license          = { :type => 'MIT', :text => 'Private module — not for distribution' }
  s.author           = { 'Developer' => 'developer@example.com' }
  s.platform         = :ios, '16.0'
  s.swift_version    = '5.0'

  # Prebuilt XCFramework produced by core/scripts/build-ios.sh
  s.vendored_frameworks = 'XCFramework/CameraGestures.xcframework'

  # Swift wrapper sources (added per stage as modules are ported)
  s.source_files = 'CameraGestures/**/*.swift'

  # Populated per stage:
  # Stage 2+: MediaPipeTasksVision for HandsRecognizing on iOS
  # s.dependency 'MediaPipeTasksVision', '~> 0.10.14'
end
