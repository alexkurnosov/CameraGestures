#include "PrefixMatcher.hpp"
#include <algorithm>
#include <map>

PrefixMatcher::PrefixMatcher(const CgPoseManifest& manifest) : manifest_(manifest) {}

void PrefixMatcher::reset() {
    observed_.clear();
}

std::optional<std::set<std::string>> PrefixMatcher::gateCloseCommitSet() const {
    auto matches = gesturesMatchingExactly(observed_);
    if (matches.empty()) return std::nullopt;
    return matches;
}

PrefixMatcher::Action PrefixMatcher::observe(int pose_id, CgClusterKind kind) {
    switch (kind) {
    case CgClusterKind::unconfirmed:
        return {Action::Kind::no_prefix};
    case CgClusterKind::idle:
        return handleIdle();
    case CgClusterKind::regular:
        return handleRegular(pose_id);
    }
    return {Action::Kind::no_prefix};
}

std::set<std::string> PrefixMatcher::commitTimerFiredSet(
        const std::set<std::string>& candidate_set) const {
    std::set<std::string> valid;
    for (const auto& gesture_id : candidate_set) {
        auto it = manifest_.gesture_templates.find(gesture_id);
        if (it == manifest_.gesture_templates.end()) continue;
        for (const auto& tmpl : it->second) {
            if (tmpl.size() >= observed_.size()) {
                auto prefix = std::vector<int>(tmpl.begin(),
                                               tmpl.begin() + observed_.size());
                if (prefix == observed_) { valid.insert(gesture_id); break; }
            }
        }
    }
    return valid.empty() ? candidate_set : valid;
}

// --------------------------------------------------------------------------
// Private
// --------------------------------------------------------------------------

PrefixMatcher::Action PrefixMatcher::handleIdle() {
    if (observed_.empty()) return {Action::Kind::idle_reset};

    auto complete = gesturesMatchingExactly(observed_);
    if (!complete.empty()) return {Action::Kind::idle_commit, complete};

    if (auto ancestor = longestCompleteAncestor()) {
        return {Action::Kind::idle_commit, *ancestor};
    }

    // Build full gesture ID set as the discard set.
    std::set<std::string> all;
    for (const auto& [id, _] : manifest_.gesture_templates) all.insert(id);
    return {Action::Kind::idle_discard, all};
}

PrefixMatcher::Action PrefixMatcher::handleRegular(int pose_id) {
    std::vector<int> candidate = observed_;
    candidate.push_back(pose_id);

    // Does any template start with `candidate`?
    std::map<std::string, bool> gestures_with_prefix; // id → has_longer
    for (const auto& [gesture_id, templates] : manifest_.gesture_templates) {
        bool any_prefix = false, any_longer = false;
        for (const auto& tmpl : templates) {
            if (tmpl.size() < candidate.size()) continue;
            auto prefix = std::vector<int>(tmpl.begin(),
                                           tmpl.begin() + candidate.size());
            if (prefix == candidate) {
                any_prefix = true;
                if (tmpl.size() > candidate.size()) any_longer = true;
            }
        }
        if (any_prefix) gestures_with_prefix[gesture_id] = any_longer;
    }

    if (gestures_with_prefix.empty()) return {Action::Kind::no_prefix};

    observed_ = candidate;

    auto exact = gesturesMatchingExactly(candidate);
    bool longer_possible = false;
    for (const auto& [_, has_longer] : gestures_with_prefix) {
        if (has_longer) { longer_possible = true; break; }
    }

    if (exact.empty()) return {Action::Kind::live_prefix};
    if (!longer_possible) return {Action::Kind::commit_now, exact};
    return {Action::Kind::start_commit_timer, exact};
}

std::set<std::string> PrefixMatcher::gesturesMatchingExactly(
        const std::vector<int>& seq) const {
    std::set<std::string> result;
    for (const auto& [gesture_id, templates] : manifest_.gesture_templates) {
        for (const auto& tmpl : templates) {
            if (tmpl == seq) { result.insert(gesture_id); break; }
        }
    }
    return result;
}

std::optional<std::set<std::string>> PrefixMatcher::longestCompleteAncestor() const {
    for (int len = static_cast<int>(observed_.size()) - 1; len >= 1; --len) {
        auto prefix = std::vector<int>(observed_.begin(), observed_.begin() + len);
        auto matches = gesturesMatchingExactly(prefix);
        if (!matches.empty()) return matches;
    }
    return std::nullopt;
}
