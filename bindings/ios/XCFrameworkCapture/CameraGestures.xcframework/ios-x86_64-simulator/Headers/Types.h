#ifndef CG_TYPES_H
#define CG_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Enums
 * ---------------------------------------------------------------------- */

typedef enum cg_handedness {
    CG_HAND_LEFT    = 0,
    CG_HAND_RIGHT   = 1,
    CG_HAND_UNKNOWN = 2
} cg_handedness;

/* Failure reasons why a HandFilm was rejected as a training example.
 * Unknown values (from newer app versions) are surfaced as CG_FILM_FAIL_UNKNOWN
 * with the raw string preserved in cg_failed_handfilm.failure_reason_raw. */
typedef enum cg_handfilm_failure_reason {
    CG_FILM_FAIL_INSUFFICIENT_IN_VIEW_DURATION = 0,
    CG_FILM_FAIL_UNKNOWN                       = 255
} cg_handfilm_failure_reason;


/* -------------------------------------------------------------------------
 * POD value types
 * ---------------------------------------------------------------------- */

typedef struct cg_point3d {
    float x, y, z;
} cg_point3d;

/* Single frame of hand landmarks. is_absent == 1 means no hand detected;
 * landmarks are all-zero placeholders. */
typedef struct cg_handshot {
    cg_point3d  landmarks[21]; /* MediaPipe hand landmark count */
    double      timestamp;     /* seconds since Unix epoch */
    cg_handedness handedness;
    int         is_absent;     /* bool: 1 = hand not detected */
} cg_handshot;

/* A sequence of handshots (opaque handle — use cg_handfilm_* functions). */
typedef struct cg_handfilm_s* cg_handfilm_ref;

typedef struct cg_gesture_prediction {
    char   gesture_id[128];
    char   gesture_name[256];
    float  confidence;
    double timestamp;
} cg_gesture_prediction;

typedef struct cg_gesture_definition {
    char id[128];
    char name[256];
    char description[512];
} cg_gesture_definition;


/* -------------------------------------------------------------------------
 * HandFilm handle API
 * ---------------------------------------------------------------------- */

cg_handfilm_ref cg_handfilm_create(double start_time);
void            cg_handfilm_destroy(cg_handfilm_ref film);

void            cg_handfilm_add_shot(cg_handfilm_ref film, const cg_handshot* shot);
void            cg_handfilm_clear(cg_handfilm_ref film);

size_t          cg_handfilm_shot_count(cg_handfilm_ref film);
/* Writes shot at index into *out. Returns 0 on out-of-range. */
int             cg_handfilm_get_shot(cg_handfilm_ref film, size_t index, cg_handshot* out);

double          cg_handfilm_start_time(cg_handfilm_ref film);
double          cg_handfilm_end_time(cg_handfilm_ref film);
double          cg_handfilm_gesture_duration(cg_handfilm_ref film);
double          cg_handfilm_in_view_duration(cg_handfilm_ref film);
size_t          cg_handfilm_in_view_frame_count(cg_handfilm_ref film);


/* -------------------------------------------------------------------------
 * GestureRegistry handle API
 * ---------------------------------------------------------------------- */

typedef struct cg_gesture_registry_s* cg_registry_ref;

/* Load from (or create at) the given JSON file path. */
cg_registry_ref cg_registry_create(const char* file_path);
void            cg_registry_destroy(cg_registry_ref registry);

/* Returns number of gestures currently registered. */
size_t          cg_registry_count(cg_registry_ref registry);

/* Writes gesture at index into *out. Returns 0 on out-of-range. */
int             cg_registry_get(cg_registry_ref registry, size_t index,
                                cg_gesture_definition* out);

/* Returns 1 if found, 0 if not. */
int             cg_registry_find_by_id(cg_registry_ref registry, const char* id,
                                       cg_gesture_definition* out);

/* Derives a slug from name and adds the gesture. Returns 1 on success,
 * 0 if the name is empty or the slug already exists. */
int             cg_registry_add(cg_registry_ref registry,
                                const char* name, const char* description,
                                cg_gesture_definition* out);

/* Removes the gesture with the given id. Returns 1 if removed, 0 if not found. */
int             cg_registry_remove(cg_registry_ref registry, const char* id);

/* Persist to disk. Returns 1 on success. */
int             cg_registry_save(cg_registry_ref registry);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CG_TYPES_H */
