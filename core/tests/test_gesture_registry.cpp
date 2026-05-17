#include <gtest/gtest.h>
#include "CameraGestures/Types.h"
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class GestureRegistryTest : public ::testing::Test {
protected:
    fs::path tmp_dir;
    std::string json_path;

    void SetUp() override {
        tmp_dir  = fs::temp_directory_path() / "cg_test_registry";
        fs::create_directories(tmp_dir);
        json_path = (tmp_dir / "gestures.json").string();
        // Start clean
        fs::remove(json_path);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir);
    }
};

TEST_F(GestureRegistryTest, EmptyRegistry) {
    cg_registry_ref r = cg_registry_create(json_path.c_str());
    EXPECT_EQ(cg_registry_count(r), 0u);
    cg_registry_destroy(r);
}

TEST_F(GestureRegistryTest, AddAndRetrieve) {
    cg_registry_ref r = cg_registry_create(json_path.c_str());
    cg_gesture_definition def{};
    EXPECT_EQ(cg_registry_add(r, "Thumbs Up", "Extend thumb upward", &def), 1);
    EXPECT_STREQ(def.id,   "thumbs_up");
    EXPECT_STREQ(def.name, "Thumbs Up");
    EXPECT_EQ(cg_registry_count(r), 1u);

    cg_gesture_definition got{};
    EXPECT_EQ(cg_registry_get(r, 0, &got), 1);
    EXPECT_STREQ(got.id, "thumbs_up");
    cg_registry_destroy(r);
}

TEST_F(GestureRegistryTest, SlugDerivation) {
    cg_registry_ref r = cg_registry_create(json_path.c_str());
    cg_gesture_definition def{};
    cg_registry_add(r, "Open Hand", "", &def);
    EXPECT_STREQ(def.id, "open_hand");
    cg_registry_destroy(r);
}

TEST_F(GestureRegistryTest, DuplicateSlugRejected) {
    cg_registry_ref r = cg_registry_create(json_path.c_str());
    EXPECT_EQ(cg_registry_add(r, "Thumbs Up", "", nullptr), 1);
    EXPECT_EQ(cg_registry_add(r, "Thumbs Up", "dup", nullptr), 0);
    EXPECT_EQ(cg_registry_count(r), 1u);
    cg_registry_destroy(r);
}

TEST_F(GestureRegistryTest, EmptyNameRejected) {
    cg_registry_ref r = cg_registry_create(json_path.c_str());
    EXPECT_EQ(cg_registry_add(r, "", "", nullptr), 0);
    EXPECT_EQ(cg_registry_count(r), 0u);
    cg_registry_destroy(r);
}

TEST_F(GestureRegistryTest, Remove) {
    cg_registry_ref r = cg_registry_create(json_path.c_str());
    cg_registry_add(r, "Thumbs Up", "", nullptr);
    EXPECT_EQ(cg_registry_count(r), 1u);
    EXPECT_EQ(cg_registry_remove(r, "thumbs_up"), 1);
    EXPECT_EQ(cg_registry_count(r), 0u);
    EXPECT_EQ(cg_registry_remove(r, "thumbs_up"), 0);  // already gone
    cg_registry_destroy(r);
}

// Persist + reload round-trip
TEST_F(GestureRegistryTest, PersistenceRoundTrip) {
    {
        cg_registry_ref r = cg_registry_create(json_path.c_str());
        cg_registry_add(r, "Thumbs Up",   "Up",   nullptr);
        cg_registry_add(r, "Peace Sign",  "Peace", nullptr);
        cg_registry_save(r);
        cg_registry_destroy(r);
    }
    // Re-open from disk
    cg_registry_ref r2 = cg_registry_create(json_path.c_str());
    EXPECT_EQ(cg_registry_count(r2), 2u);
    cg_gesture_definition d0{}, d1{};
    cg_registry_get(r2, 0, &d0);
    cg_registry_get(r2, 1, &d1);
    EXPECT_STREQ(d0.id, "thumbs_up");
    EXPECT_STREQ(d1.id, "peace_sign");
    cg_registry_destroy(r2);
}

// Load a gestures.json that was produced by the iOS app (same format)
TEST_F(GestureRegistryTest, LoadIOSFormat) {
    const char* json_content = R"([
      {"id":"wave","name":"Wave","description":"Wave hello"},
      {"id":"fist","name":"Fist","description":"Closed fist"}
    ])";
    std::ofstream f(json_path);
    f << json_content;
    f.close();

    cg_registry_ref r = cg_registry_create(json_path.c_str());
    EXPECT_EQ(cg_registry_count(r), 2u);
    cg_gesture_definition def{};
    EXPECT_EQ(cg_registry_find_by_id(r, "fist", &def), 1);
    EXPECT_STREQ(def.name, "Fist");
    cg_registry_destroy(r);
}

TEST_F(GestureRegistryTest, FindByIdMissing) {
    cg_registry_ref r = cg_registry_create(json_path.c_str());
    EXPECT_EQ(cg_registry_find_by_id(r, "nonexistent", nullptr), 0);
    cg_registry_destroy(r);
}
