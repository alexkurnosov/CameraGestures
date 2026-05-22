// CameraGestures Android binding library.
// Plugin versions are declared in the root project's pluginManagement block
// (apps/demo-android/build.gradle.kts) — do NOT repeat them here.
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace  = "com.cameragestures"
    compileSdk = 34
    ndkVersion = "27.3.13750724"

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

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    // Bundle hand_landmarker.task from the shared core assets directory.
    sourceSets["main"].assets.srcDirs("src/main/assets", "../../core/assets")
}

dependencies {
    // Stage 3: MediaPipe tasks-vision for hand landmark detection on Android.
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}
