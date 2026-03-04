Pod::Spec.new do |spec|
  spec.name             = 'HandGestureRecognizingFramework'
  spec.version          = '0.1.0'
  spec.summary          = 'Production-ready gesture recognition module'
  spec.description      = <<-DESC
                       HandGestureRecognizing orchestrates hand tracking and gesture classification
                       to provide real-time gesture recognition capabilities for external applications.
                       Combines HandsRecognizing and GestureModel modules with production-ready APIs.
                       DESC

  spec.homepage         = 'https://github.com/yourname/CameraGestures'
  spec.license          = { :type => 'MIT', :text => 'Private module - not for distribution' }
  spec.author           = { 'Developer' => 'developer@example.com' }

  spec.platform         = :ios, '15.0'
  spec.swift_version    = '5.0'

  # Source location (local path)
  spec.source           = { :path => '.' }

  # Source files
  spec.source_files     = 'HandGestureRecognizing/HandGestureRecognizing/**/*.swift'
  
  # Dependencies on local pods
  spec.dependency 'HandsRecognizingModule', '0.1.0'
  spec.dependency 'HandGestureTypes', '0.1.0'
  spec.dependency 'GestureModelModule', '0.1.0'
  
  # Framework settings
  spec.requires_arc     = true
  spec.frameworks       = 'UIKit'
end
