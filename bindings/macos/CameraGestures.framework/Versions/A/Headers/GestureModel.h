#ifndef CG_GESTURE_MODEL_H
#define CG_GESTURE_MODEL_H

#include "Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Opaque handle
 * ---------------------------------------------------------------------- */

typedef struct cg_gesture_model_s* cg_gesture_model_ref;

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/* Load the Phase-3 gesture MLP with the model's own class list. PREFERRED.
 * tflite_path : path to the .tflite model file produced by the server trainer.
 * gesture_ids : the class list the server shipped with this model, in the
 *               model's output order — i.e. gesture_ids[i] names output i.
 *               Taken verbatim: not sorted, nothing appended. The server is the
 *               only party that knows how its own model was trained.
 * count       : number of entries; must equal the model's output width.
 * Returns NULL on failure (file not found, dim mismatch, TFLite not compiled in).
 *
 * Clients obtain the list from the `gesture_ids.json` sidecar written next to
 * the downloaded model. Use this rather than cg_gesture_model_load whenever the
 * sidecar is present. */
cg_gesture_model_ref cg_gesture_model_load_with_ids(const char* tflite_path,
                                                     const char* const* gesture_ids,
                                                     int count);

/* Load the Phase-3 gesture MLP, deriving the class list locally. LEGACY.
 * tflite_path   : path to the .tflite model file produced by the server trainer.
 * registry_path : path to a gestures.json GestureRegistry file. The library
 *                 reads all gesture IDs from it, appends "_none", sorts them
 *                 alphabetically, and requires the result to match the model's
 *                 output class order.
 * Returns NULL on failure (file not found, dim mismatch, TFLite not compiled in).
 *
 * WARNING: this reconstructs, rather than reads, a contract the server owns.
 * It holds only while the server trains exactly the registry's gestures plus
 * "_none". When the server dropped the "_none" class on 2026-09-09 this
 * assumption broke and every client stopped predicting gestures entirely —
 * silently, because runPhase3 only returns false on a width mismatch. Prefer
 * cg_gesture_model_load_with_ids; keep this only for callers with no sidecar. */
cg_gesture_model_ref cg_gesture_model_load(const char* tflite_path,
                                            const char* registry_path);

void cg_gesture_model_destroy(cg_gesture_model_ref model);

/* Load the Phase-2 pose MLP alongside its manifest JSON.
 * Must be called after cg_gesture_model_load on the same ref.
 * Returns 1 on success, 0 on failure. */
int cg_gesture_model_load_pose(cg_gesture_model_ref model,
                                const char* tflite_path,
                                const char* manifest_path);

/* -------------------------------------------------------------------------
 * Phase-3 classification (full HandFilm → gesture)
 * ---------------------------------------------------------------------- */

/* Classify a full HandFilm. Writes the best prediction to *out.
 * threshold : minimum softmax probability to report (use 0.0 to always report best).
 * Returns 1 on success (out written), 0 if model not loaded / below threshold. */
int cg_gesture_model_classify(cg_gesture_model_ref model,
                               cg_handfilm_ref      film,
                               float                threshold,
                               cg_gesture_prediction* out);

/* Top-k variant. out must point to an array of at least max_out elements.
 * Returns the number of predictions written (≤ k, ≤ max_out, ≤ n_gestures). */
int cg_gesture_model_classify_topk(cg_gesture_model_ref    model,
                                    cg_handfilm_ref          film,
                                    int                      k,
                                    float                    threshold,
                                    cg_gesture_prediction*   out,
                                    int                      max_out);

/* Masked-argmax: only consider gesture IDs listed in candidate_ids.
 * candidate_ids: array of C strings, length n_candidates.
 * Returns 1 on success (out written), 0 otherwise. */
int cg_gesture_model_classify_restricted(cg_gesture_model_ref  model,
                                          cg_handfilm_ref        film,
                                          const char* const*     candidate_ids,
                                          int                    n_candidates,
                                          float                  threshold,
                                          cg_gesture_prediction* out);

/* -------------------------------------------------------------------------
 * Phase-2 classification (single frame → pose cluster)
 * ---------------------------------------------------------------------- */

/* Predict the pose cluster for a single HandShot using the pose MLP.
 * pose_id_out     : receives the integer cluster ID.
 * confidence_out  : receives the softmax probability of the winning class.
 * Returns 1 on success, 0 if pose model not loaded. */
int cg_gesture_model_predict_pose(cg_gesture_model_ref model,
                                   const cg_handshot*   shot,
                                   int*                 pose_id_out,
                                   float*               confidence_out);

/* Returns the number of pose classes the loaded pose MLP knows (0 if not loaded). */
int cg_gesture_model_pose_class_count(cg_gesture_model_ref model);

/* Like cg_gesture_model_predict_pose but also fills scores_out with softmax
 * probabilities for ALL pose classes in class_labels order (same ordering used
 * by the manifest's class_labels array).
 * scores_capacity must be >= cg_gesture_model_pose_class_count().
 * scores_out and n_scores_out may be NULL.
 * Returns 1 on success, 0 otherwise. */
int cg_gesture_model_predict_pose_all_scores(cg_gesture_model_ref model,
                                              const cg_handshot*   shot,
                                              int*                 pose_id_out,
                                              float*               confidence_out,
                                              float*               scores_out,
                                              int                  scores_capacity,
                                              int*                 n_scores_out);

/* -------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------- */

/* Set GEOM_COEF (default 1.0). Must match the value used during training.
 * Read from the .meta.json file alongside the .tflite (key "geom_coef"). */
void cg_gesture_model_set_geom_coef(cg_gesture_model_ref model, float coef);

/* Number of supported gesture IDs (including "_none"). */
int cg_gesture_model_gesture_count(cg_gesture_model_ref model);

#ifdef __cplusplus
}
#endif

#endif /* CG_GESTURE_MODEL_H */
