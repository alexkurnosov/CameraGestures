plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace  = "com.cameragestures.demo"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.cameragestures.demo"
        minSdk        = 24
        targetSdk     = 34
        versionCode   = 1
        versionName   = "1.0"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.11"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    // gesture_model.tflite + gestures.json go here (user must copy them before building).
    sourceSets["main"].assets.srcDirs("src/main/assets")
}

// MediaPipe brings in guava:27.0.1-android which already contains ListenableFuture.
// Exclude the standalone listenablefuture artifact everywhere so there is only one
// copy of the class on the classpath (the one inside guava).
configurations.all {
    exclude(group = "com.google.guava", module = "listenablefuture")
}

dependencies {
    implementation(project(":cameragestures"))

    // Compose BOM pins all Compose library versions.
    val composeBom = platform("androidx.compose:compose-bom:2024.02.00")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")

    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    // CameraX
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")

    // ListenableFuture (needed to call ProcessCameraProvider.getInstance().await()).
    // The standalone listenablefuture artifact is excluded above; guava provides the class.
    implementation("com.google.guava:guava:27.0.1-android")
    implementation("androidx.concurrent:concurrent-futures-ktx:1.1.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Accompanist — runtime permissions helper for Compose
    implementation("com.google.accompanist:accompanist-permissions:0.34.0")
}
