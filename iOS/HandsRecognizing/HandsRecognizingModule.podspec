Pod::Spec.new do |spec|
  spec.name             = 'HandsRecognizingModule'
  spec.version          = '0.1.0'
  spec.summary          = 'Real-time hand detection and coordinate extraction using MediaPipe'
  spec.description      = <<-DESC
                       HandsRecognizing module provides real-time hand detection and tracking 
                       capabilities using MediaPipe Hands. Captures hand landmarks and generates 
                       HandShot and HandFilm data structures for gesture recognition.
                       DESC

  spec.homepage         = 'https://github.com/yourname/CameraGestures'
  spec.license          = { :type => 'MIT', :text => 'Private module - not for distribution' }
  spec.author           = { 'Developer' => 'developer@example.com' }

  spec.platform         = :ios, '15.0'
  spec.swift_version    = '5.0'

  # Source location (local path)
  spec.source           = { :path => '.' }

  spec.source_files = 'HandsRecognizing/HandsRecognizing/**/*.swift'
  spec.resources = 'hand_landmarker.task'

  # Dependencies
  spec.dependency 'MediaPipeTasksVision', '0.10.14'
  spec.dependency 'HandGestureTypes', '0.1.0'
  
  # Framework settings
  spec.requires_arc     = true
  spec.frameworks       = 'AVFoundation', 'UIKit'
end
