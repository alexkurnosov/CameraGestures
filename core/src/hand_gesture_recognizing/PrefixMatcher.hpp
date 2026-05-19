#pragma once
#include "PoseManifest.hpp"
#include <vector>
#include <string>
#include <set>
#include <optional>

// Mirror of iOS HandGestureRecognizing/PrefixMatcher.swift.
// Pure Phase-2 prefix-matching state machine — no threading, no timers, no I/O.
// Feed pose predictions one at a time (on each detected hold); read the returned Action.
// HandGestureRecognizing owns the T_commit timer and calls commitTimerFiredSet() when it fires.

class PrefixMatcher {
public:
    struct Action {
        enum class Kind {
            no_prefix,
            live_prefix,
            commit_now,
            start_commit_timer,
            idle_reset,
            idle_discard,
            idle_commit,
        };
        Kind             kind         = Kind::no_prefix;
        std::set<std::string> candidate_set; // populated for commit_* and idle_commit/idle_discard
    };

    explicit PrefixMatcher(const CgPoseManifest& manifest);

    void reset();

    // Returns candidate set if `observed` has any complete template match.
    // Used on gate-close commit path.
    std::optional<std::set<std::string>> gateCloseCommitSet() const;

    // Feed one pose prediction; returns the action to take.
    Action observe(int pose_id, CgClusterKind kind);

    const std::vector<int>& observedSequence() const { return observed_; }

    // Called when T_commit timer fires; returns the validated candidate set.
    std::set<std::string> commitTimerFiredSet(const std::set<std::string>& candidate_set) const;

private:
    const CgPoseManifest& manifest_;
    std::vector<int>      observed_;

    std::set<std::string> gesturesMatchingExactly(const std::vector<int>& seq) const;
    std::optional<std::set<std::string>> longestCompleteAncestor() const;
    Action handleIdle();
    Action handleRegular(int pose_id);
};
