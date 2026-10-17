/*
 * session_video_check — where a sharp movement sits in a session's video.
 *
 * Session capture Stage 4. Reads a .cgsession that has a video and prints, for
 * the camera frames around a clap, each frame's video time next to what the
 * recorded landmarks show. Extract the video frames at those times (the tool
 * prints an ffmpeg command) and compare: the two agree when the video and the
 * shot record are aligned.
 *
 * With two hands recorded, the clap is the frame where the hands are closest.
 * With one hand, the tool marks the frame where that hand moves fastest.
 *
 * The recognizer's raw energy is not used. With two hands in view the shots
 * reach the gate as one interleaved stream, so its energy is the distance
 * between the hands and says nothing about when they moved.
 *
 *   session_video_check <bundle.cgsession> [--from S] [--to S] [--rows N]
 *
 * --from / --to limit the search to a window, in seconds from the first shot.
 * --rows is how many frames to print on each side of the marked one (default 8).
 */

#include "CameraGestures/CameraGestures.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<cg_session_shot> readTrack(cg_session_reader_ref rd, int track) {
    std::vector<cg_session_shot> shots(cg_session_reader_shot_count(rd, track));
    shots.resize(cg_session_reader_read_shots(rd, track, 0, shots.size(), shots.data()));
    return shots;
}

// Mean distance in the image plane between the same landmark of two shots.
double meanDistance(const cg_handshot& a, const cg_handshot& b) {
    double sum = 0.0;
    for (int i = 0; i < 21; ++i) {
        sum += std::hypot(a.landmarks[i].x - b.landmarks[i].x, a.landmarks[i].y - b.landmarks[i].y);
    }
    return sum / 21.0;
}

double median(std::vector<double>::const_iterator first, std::vector<double>::const_iterator last) {
    std::vector<double> sorted(first, last);
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    const char* bundle = nullptr;
    double from = 0.0, to = 1e18;
    long   around = 8;
    for (int i = 1; i < argc; ++i) {
        const bool has_value = i + 1 < argc;
        if (!std::strcmp(argv[i], "--from") && has_value)      from   = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--to") && has_value)   to     = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--rows") && has_value) around = std::atol(argv[++i]);
        else if (argv[i][0] != '-' && !bundle)                 bundle = argv[i];
        else bundle = nullptr, i = argc;
    }
    if (!bundle) {
        std::fprintf(stderr, "usage: session_video_check <bundle.cgsession> [--from S] [--to S] [--rows N]\n");
        return 2;
    }

    char error[512];
    cg_session_reader_ref rd = cg_session_reader_open(bundle, error, sizeof(error));
    if (!rd) {
        std::fprintf(stderr, "cannot open %s: %s\n", bundle, error);
        return 1;
    }

    cg_session_provenance prov{};
    cg_session_reader_get_provenance(rd, &prov);
    const cg_session_video_info& v = prov.video;
    if (!v.has_video) {
        std::fprintf(stderr, "the session has no video\n");
        return 1;
    }
    std::printf("video      %dx%d, rotation %d, %s\n", v.width, v.height, v.rotation_degrees,
                v.mirrored ? "mirrored" : "not mirrored");
    if (!v.has_anchor) {
        std::fprintf(stderr, "the session has no video anchor (no video frame was written)\n");
        return 1;
    }
    std::printf("anchor     pts %.6f  timestamp %.3f\n", v.anchor_pts, v.anchor_timestamp);

    double start = 0.0, end = 0.0;
    if (!cg_session_reader_time_range(rd, &start, &end)) {
        std::fprintf(stderr, "the session has no shots\n");
        return 1;
    }
    std::printf("session    %.1f s, %s\n", end - start,
                cg_session_reader_is_complete(rd) ? "complete" : "interrupted");

    const std::vector<cg_session_shot> hand0 = readTrack(rd, 0), hand1 = readTrack(rd, 1);
    if (hand0.empty()) {
        std::fprintf(stderr, "the session has no shots of hand 0\n");
        return 1;
    }

    // The shot timestamp is taken when the frame reaches the app, the PTS when
    // it was captured. If the gap between them grows over the session, the two
    // clocks drift apart.
    std::vector<double> gaps;
    for (const cg_session_shot& s : hand0) {
        if (s.has_pts) gaps.push_back((s.shot.timestamp - v.anchor_timestamp) - (s.pts - v.anchor_pts));
    }
    if (gaps.size() >= 200) {
        std::printf("clocks     timestamp minus PTS, relative to the anchor, median of 100 shots: "
                    "first %+.1f ms, last %+.1f ms\n",
                    median(gaps.begin(), gaps.begin() + 100) * 1e3,
                    median(gaps.end() - 100, gaps.end()) * 1e3);
    }

    // One row per camera frame that has hand 0.
    struct Frame { double t, video, distance, speed0, speed1; };
    std::vector<Frame> frames;
    size_t j = 0;
    bool   two_hands = false;
    for (size_t i = 0; i < hand0.size(); ++i) {
        const cg_session_shot& a = hand0[i];
        Frame f{a.shot.timestamp, a.has_pts ? a.pts - v.anchor_pts : -1.0, -1.0, -1.0, -1.0};
        if (i > 0) f.speed0 = meanDistance(a.shot, hand0[i - 1].shot);
        while (j < hand1.size() && hand1[j].shot.timestamp < f.t) ++j;
        if (j < hand1.size() && hand1[j].shot.timestamp == f.t) {
            f.distance = meanDistance(a.shot, hand1[j].shot);
            if (j > 0) f.speed1 = meanDistance(hand1[j].shot, hand1[j - 1].shot);
            two_hands = true;
        }
        frames.push_back(f);
    }

    long marked = -1;
    for (size_t i = 0; i < frames.size(); ++i) {
        const Frame& f = frames[i];
        const double rel = f.t - start;
        const double value = two_hands ? f.distance : f.speed0;
        if (rel < from || rel > to || value < 0 || f.video < 0) continue;
        const double best = marked < 0 ? 0.0 : (two_hands ? frames[marked].distance : frames[marked].speed0);
        if (marked < 0 || (two_hands ? value < best : value > best)) marked = static_cast<long>(i);
    }
    if (marked < 0) {
        std::fprintf(stderr, "no frame with a video time in the window\n");
        return 1;
    }

    // The measured interval, not the nominal one: in dim light the camera slows down.
    std::vector<double> steps;
    for (size_t i = 1; i < frames.size(); ++i) {
        if (frames[i].video >= 0 && frames[i - 1].video >= 0) steps.push_back(frames[i].video - frames[i - 1].video);
    }
    const double frame = steps.empty() ? 1.0 / 30.0 : median(steps.begin(), steps.end());
    const char*  what  = two_hands ? "hands closest" : "fastest";
    std::printf("frame      %.1f ms\n\n", frame * 1e3);
    std::printf("Distances are in normalised frame units, averaged over the 21 landmarks.\n\n");
    std::printf(" t_rel_s   video_s  hands_apart  moved_0  moved_1\n");
    auto cell = [](double value, const char* format = "%.4f") {
        char text[16] = "    -  ";
        if (value >= 0) std::snprintf(text, sizeof(text), format, value);
        return std::string(text);
    };
    const long lo = marked - around < 0 ? 0 : marked - around;
    const long hi = marked + around >= static_cast<long>(frames.size())
                  ? static_cast<long>(frames.size()) - 1 : marked + around;
    for (long i = lo; i <= hi; ++i) {
        const Frame& f = frames[i];
        std::printf("%8.3f  %8s  %11s  %7s  %7s%s%s\n", f.t - start,
                    cell(f.video, "%.3f").c_str(),
                    cell(f.distance).c_str(), cell(f.speed0).c_str(), cell(f.speed1).c_str(),
                    i == marked ? "   <- " : "", i == marked ? what : "");
    }

    // Half a frame of slack on each side, so rounding does not cut a frame off.
    const double at = frames[marked].video;
    const double a = at - 3.5 * frame, b = at + 3.5 * frame;
    std::printf("\nThe marked frame is at video time %.3f s. Frames from 3 before to 3 after it:\n\n", at);
    std::printf("  ffmpeg -hide_banner -y -i '%s/video.mov' "
                "-vf \"select='between(t,%.3f,%.3f)',showinfo\" -vsync 0 clap_%%02d.png "
                "2>&1 | grep -o 'pts_time:[0-9.]*'\n", bundle, a < 0 ? 0.0 : a, b);
    cg_session_reader_close(rd);
    return 0;
}
