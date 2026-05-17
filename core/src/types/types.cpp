#include "CameraGestures/Types.h"
#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>

/* -------------------------------------------------------------------------
 * Internal HandFilm struct
 * ---------------------------------------------------------------------- */

struct cg_handfilm_s {
    std::vector<cg_handshot> shots;
    double start_time;

    explicit cg_handfilm_s(double t) : start_time(t) {}
};

/* -------------------------------------------------------------------------
 * HandFilm handle API
 * ---------------------------------------------------------------------- */

cg_handfilm_ref cg_handfilm_create(double start_time) {
    return new cg_handfilm_s(start_time);
}

void cg_handfilm_destroy(cg_handfilm_ref film) {
    delete film;
}

void cg_handfilm_add_shot(cg_handfilm_ref film, const cg_handshot* shot) {
    if (film && shot) film->shots.push_back(*shot);
}

void cg_handfilm_clear(cg_handfilm_ref film) {
    if (film) film->shots.clear();
}

size_t cg_handfilm_shot_count(cg_handfilm_ref film) {
    return film ? film->shots.size() : 0;
}

int cg_handfilm_get_shot(cg_handfilm_ref film, size_t index, cg_handshot* out) {
    if (!film || index >= film->shots.size() || !out) return 0;
    *out = film->shots[index];
    return 1;
}

double cg_handfilm_start_time(cg_handfilm_ref film) {
    return film ? film->start_time : 0.0;
}

double cg_handfilm_end_time(cg_handfilm_ref film) {
    if (!film || film->shots.empty()) return film ? film->start_time : 0.0;
    return film->shots.back().timestamp;
}

double cg_handfilm_gesture_duration(cg_handfilm_ref film) {
    return cg_handfilm_end_time(film) - cg_handfilm_start_time(film);
}

/* Mirrors iOS inViewDuration: sum of intervals between consecutive non-absent frames. */
double cg_handfilm_in_view_duration(cg_handfilm_ref film) {
    if (!film) return 0.0;
    std::vector<double> visible_ts;
    for (const auto& s : film->shots) {
        if (!s.is_absent) visible_ts.push_back(s.timestamp);
    }
    if (visible_ts.size() < 2) return 0.0;
    double total = 0.0;
    for (size_t i = 1; i < visible_ts.size(); ++i) {
        total += visible_ts[i] - visible_ts[i - 1];
    }
    return total;
}

size_t cg_handfilm_in_view_frame_count(cg_handfilm_ref film) {
    if (!film) return 0;
    size_t count = 0;
    for (const auto& s : film->shots) {
        if (!s.is_absent) ++count;
    }
    return count;
}
