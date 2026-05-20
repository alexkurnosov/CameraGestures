// Stage 3: HandsRecognizing Android binding scaffold.
plugins {
    id("com.android.library") version "8.3.0"
    id("org.jetbrains.kotlin.android") version "1.9.0"
}

android {
    namespace  = "com.cameragestures"
    compileSdk = 34

    defaultConfig {
        minSdk = 24

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                // Expose prebuilt C ABI static lib path to the JNI bridge build.
                arguments += "-DCAMERAGESTURES_PREBUILT_DIR=\${projectDir}/src/main/cpp/prebuilt"
            }
        }

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path    = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // Bundle hand_landmarker.task from the shared core assets directory.
    sourceSets["main"].assets.srcDirs("src/main/assets", "../../core/assets")
}

dependencies {
    // Stage 3: MediaPipe tasks-vision for hand landmark detection on Android.
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}
