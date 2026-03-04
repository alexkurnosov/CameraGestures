Pod::Spec.new do |spec|
  spec.name             = 'GestureModelModule'
  spec.version          = '0.1.0'
  spec.summary          = 'Neural network abstraction layer for gesture classification'
  spec.description      = <<-DESC
                       GestureModel provides a unified API for different machine learning backends
                       to classify gestures from HandFilm sequences. Currently supports mock 
                       implementation with planned TensorFlow backend integration.
                       DESC

  spec.homepage         = 'https://github.com/yourname/CameraGestures'
  spec.license          = { :type => 'MIT', :text => 'Private module - not for distribution' }
  spec.author           = { 'Developer' => 'developer@example.com' }

  spec.platform         = :ios, '15.0'
  spec.swift_version    = '5.0'

  # Source location (local path)
  spec.source           = { :path => '.' }

  # Source files
  spec.source_files     = 'GestureModel/GestureModel/**/*.swift'
  
  # Dependencies
  spec.dependency 'HandGestureTypes', '0.1.0'
  spec.dependency 'TensorFlowLiteSwift', '~> 2.13.0'
  
  # Framework settings
  spec.requires_arc     = true
end
