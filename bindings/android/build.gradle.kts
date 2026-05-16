// Stage 0 placeholder — fleshed out in Stage 3 (HandsRecognizing Android).
plugins {
    id("com.android.library") version "8.3.0"
    id("org.jetbrains.kotlin.android") version "1.9.0"
}

android {
    namespace = "com.cameragestures"
    compileSdk = 34
    defaultConfig {
        minSdk = 24
    }
}

dependencies {
    // Populated per stage:
    // Stage 3+: implementation("com.google.mediapipe:tasks-vision:0.10.14")
}
