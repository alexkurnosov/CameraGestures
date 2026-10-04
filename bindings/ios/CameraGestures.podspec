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
  #   It comes in two variants, one per subspec below.
  # TensorFlowLiteC.xcframework: vendored TFLite runtime (statically linked,
  #   no TFLite Swift pod needed).
  #
  # The Standard and SessionCapture subspecs exclude each other: each ships its
  # own build of the same static library, so selecting both fails at link time
  # with duplicate symbols.
  #
  #   pod 'CameraGestures'                  the standard library, no capture code
  #   pod 'CameraGestures/SessionCapture'   adds session capture (SessionRecorder)
  s.default_subspec = 'Standard'

  # Everything the two variants share.
  s.subspec 'Core' do |ss|
    # Swift wrapper sources.
    ss.source_files = 'CameraGestures/**/*.swift'

    ss.vendored_frameworks = '../../core/third_party/tflite/ios/TensorFlowLiteC.xcframework'

    # hand_landmarker.task must be bundled so HandsRecognizingConfig can load it at runtime.
    # resource_bundles is used instead of resources so CocoaPods reliably creates a
    # CameraGesturesAssets.bundle build phase regardless of linkage mode.
    ss.resource_bundles = {
      'CameraGesturesAssets' => ['hand_landmarker.task']
    }

    # Stage 3+: HandsRecognizing uses MediaPipeTasksVision for iOS landmark detection.
    ss.dependency 'MediaPipeTasksVision', '0.10.14'
  end

  s.subspec 'Standard' do |ss|
    ss.dependency 'CameraGestures/Core'
    ss.vendored_frameworks = 'XCFramework/CameraGestures.xcframework'
  end

  s.subspec 'SessionCapture' do |ss|
    ss.dependency 'CameraGestures/Core'
    # Built by `core/scripts/build-ios.sh --capture`.
    ss.vendored_frameworks = 'XCFrameworkCapture/CameraGestures.xcframework'
    # The pod's Swift capture API is behind `#if CG_SESSION_CAPTURE`. The app
    # target gets the same condition so it can gate its own capture UI.
    capture_flag = { 'SWIFT_ACTIVE_COMPILATION_CONDITIONS' => '$(inherited) CG_SESSION_CAPTURE' }
    ss.pod_target_xcconfig  = capture_flag
    ss.user_target_xcconfig = capture_flag
  end
end
