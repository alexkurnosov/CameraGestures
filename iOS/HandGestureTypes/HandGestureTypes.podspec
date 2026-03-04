Pod::Spec.new do |spec|
  spec.name             = 'HandGestureTypes'
  spec.version          = '0.1.0'
  spec.summary          = 'Core data types and structures for hand gesture recognition'
  spec.description      = <<-DESC
                       Provides fundamental data structures and types used across the CameraGestures 
                       system including HandShot, HandFilm, Point3D, and gesture recognition enums.
                       DESC

  spec.homepage         = 'https://github.com/yourname/CameraGestures'
  spec.license          = { :type => 'MIT', :text => 'Private module - not for distribution' }
  spec.author           = { 'Developer' => 'developer@example.com' }

  spec.platform         = :ios, '15.0'
  spec.swift_version    = '5.0'

  # Source location (local path)
  spec.source           = { :path => '.' }

  # Source files
  spec.source_files     = 'HandGestureTypes/**/*.swift'
  
  # Framework settings
  spec.requires_arc     = true
end
