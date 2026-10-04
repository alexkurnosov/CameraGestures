// swift-tools-version: 5.7
// Runs the pure-Swift tests of the iOS binding on a Mac, with `swift test` in
// this directory. The library itself is built as a CocoaPod
// (CameraGestures.podspec); this package only covers the sources that need
// neither the C core nor a device.

import PackageDescription

let package = Package(
    name: "CameraGesturesBindingTests",
    targets: [
        .target(
            name: "CooldownQueue",
            path: "CameraGestures",
            sources: ["CooldownQueue.swift"]
        ),
        .testTarget(
            name: "CooldownQueueTests",
            dependencies: ["CooldownQueue"],
            path: "Tests/CooldownQueueTests"
        ),
    ]
)
