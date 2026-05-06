import Foundation
import GestureModelModule

/// Pure Phase 2 prefix-matching state machine.
///
/// No threading, no timers — feed pose predictions one at a time (on each detected
/// hold), read the returned Action. HandGestureRecognizing owns the T_commit timer
/// and calls `commitCurrentMatch()` when it fires.
///
/// Implements plan §Phase 2 Runtime flow rules for regular, idle, and
/// unconfirmed poses.
public final class PrefixMatcher {

    // MARK: - Action

    public enum Action {
        /// No template starts with `observed` — discard and reset the gate.
        case noPrefix
        /// `observed` is a live prefix only; no complete match yet — keep buffering.
        case livePrefix
        /// Complete match, no longer prefix possible — commit with `candidateSet`.
        case commitNow(candidateSet: Set<String>)
        /// Complete match exists but longer prefixes are still possible —
        /// start T_commit; commit `candidateSet` if it fires.
        case startCommitTimer(candidateSet: Set<String>)
        /// Idle pose on empty `observed` — reset gate, keep watching.
        case idleReset
        /// Idle pose with no complete match and no complete ancestor — discard.
        case idleDiscard(candidateSet: Set<String>)
        /// Idle pose on complete match — commit immediately.
        case idleCommit(candidateSet: Set<String>)
    }

    // MARK: - State

    private let manifest: PoseManifest
    public private(set) var observedSequence: [Int] = []

    public init(manifest: PoseManifest) {
        self.manifest = manifest
    }

    // MARK: - Public API

    public func reset() {
        observedSequence.removeAll()
    }

    /// Returns the candidate set if `observed` has any complete template match,
    /// for use on gate-close (the gate-close commit path).
    public func gateCloseCommitSet() -> Set<String>? {
        let matches = gesturesMatchingExactly(observedSequence)
        return matches.isEmpty ? nil : matches
    }

    /// Feed one pose prediction and get back the action HandGestureRecognizing should take.
    public func observe(poseId: Int, kind: ClusterKind) -> Action {
        switch kind {
        case .unconfirmed:
            // Unconfirmed clusters are rejected at runtime — same as sub-τ confidence.
            return .noPrefix

        case .idle:
            return handleIdle()

        case .regular:
            return handleRegular(poseId: poseId)
        }
    }

    // MARK: - Private — idle path

    private func handleIdle() -> Action {
        if observedSequence.isEmpty {
            return .idleReset
        }
        let complete = gesturesMatchingExactly(observedSequence)
        if !complete.isEmpty {
            return .idleCommit(candidateSet: complete)
        }
        // Live prefix with no complete match — find longest complete ancestor
        if let ancestor = longestCompleteAncestor() {
            return .idleCommit(candidateSet: ancestor)
        }
        return .idleDiscard(candidateSet: Set(manifest.gestureTemplates.keys))
    }

    // MARK: - Private — regular path

    private func handleRegular(poseId: Int) -> Action {
        let candidate = observedSequence + [poseId]

        // Does any template (across all gestures) start with this candidate?
        let gesturesWithPrefix = manifest.gestureTemplates.filter { (_, templates) in
            templates.contains { template in
                template.count >= candidate.count &&
                Array(template.prefix(candidate.count)) == candidate
            }
        }

        guard !gesturesWithPrefix.isEmpty else {
            return .noPrefix
        }

        // Commit candidate as the new observed
        observedSequence = candidate

        // Which gestures have an exact template match?
        let exactMatches = gesturesMatchingExactly(candidate)
        let longerPossible = gesturesWithPrefix.values.contains { templates in
            templates.contains { $0.count > candidate.count }
        }

        if exactMatches.isEmpty {
            // Only live prefixes — keep waiting
            return .livePrefix
        } else if !longerPossible {
            // Complete match, no longer prefix possible — commit immediately
            return .commitNow(candidateSet: exactMatches)
        } else {
            // Complete match but longer still possible — start T_commit timer
            return .startCommitTimer(candidateSet: exactMatches)
        }
    }

    // MARK: - Private helpers

    private func gesturesMatchingExactly(_ sequence: [Int]) -> Set<String> {
        Set(manifest.gestureTemplates.compactMap { (gestureId, templates) in
            templates.contains { $0 == sequence } ? gestureId : nil
        })
    }

    /// Longest strict prefix of `observedSequence` that is a complete template match.
    private func longestCompleteAncestor() -> Set<String>? {
        for length in stride(from: observedSequence.count - 1, through: 1, by: -1) {
            let prefix = Array(observedSequence.prefix(length))
            let matches = gesturesMatchingExactly(prefix)
            if !matches.isEmpty { return matches }
        }
        return nil
    }

    // MARK: - Commit

    /// Called by HandGestureRecognizing when the T_commit timer fires.
    /// Returns the candidate set that was pending at timer start — the caller
    /// is responsible for not calling this if the gate has already closed.
    public func commitTimerFiredSet(for candidateSet: Set<String>) -> Set<String> {
        // Re-validate: gestures in the set that still have a template prefixed by observed
        let valid = candidateSet.filter { gestureId in
            guard let templates = manifest.gestureTemplates[gestureId] else { return false }
            return templates.contains { template in
                Array(template.prefix(observedSequence.count)) == observedSequence
            }
        }
        return valid.isEmpty ? candidateSet : valid
    }
}
