#ifndef CAMERA_GESTURES_H
#define CAMERA_GESTURES_H

#include "Types.h"
#include "HandsRecognizing.h"
#include "GestureModel.h"
#include "HandGestureRecognizing.h"
/* Present only in the capture variant of the library; the standard variant
 * ships without this header and without the code behind it. */
#if __has_include("SessionCapture.h")
#include "SessionCapture.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the library version string, e.g. "0.1.0". */
const char* cg_version(void);

#ifdef __cplusplus
}
#endif

#endif /* CAMERA_GESTURES_H */
