pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "CameraGesturesDemo"

include(":app")

// The CameraGestures binding library (HandsRecognizing, GestureModel, HandGestureRecognizing).
include(":cameragestures")
project(":cameragestures").projectDir = file("../../bindings/android")
