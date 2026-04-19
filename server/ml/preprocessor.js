/**
 * preprocessor.js — single source of truth for hand gesture feature extraction.
 *
 * Runs on:
 *   - Server: via py_mini_racer (Python)
 *   - iOS:    via JavaScriptCore (Swift)
 *
 * Input format (HandFilm JSON):
 *   {
 *     "frames": [
 *       {
 *         "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}, ...],  // 21 items
 *         "timestamp": 1234567890.123,
 *         "left_or_right": "right"
 *       },
 *       ...
 *     ],
 *     "start_time": 1234567890.0
 *   }
 */

var TARGET_FRAMES = 60;
var LANDMARKS = 21;
var COORDS = 3;
var FEATURES_PER_FRAME = LANDMARKS * COORDS * 2; // 126
var COORDS_PER_FRAME = LANDMARKS * COORDS;        // 63

/**
 * Build the (TARGET_FRAMES × 126) feature matrix.
 * Returns a flat Float64 array of length TARGET_FRAMES * 126.
 *
 * Per-frame normalisation pipeline (makes features invariant to hand pose):
 *   1. Handedness flip   — mirror x for left hands so they match right hands.
 *   2. Wrist-relative    — subtract wrist position (translation-invariant).
 *   3. Scale-normalised  — divide by |wrist → middle_MCP| (size-invariant).
 *   4. Rotation-aligned  — rotate into a hand-local frame built from palm
 *                          knuckles (forearm-orientation-invariant).
 *
 * Columns 0–62:  normalised landmark coords.
 * Columns 63–125: frame-to-frame velocity of those coords (zero for first frame).
 */
function featureMatrix(handFilm) {
    var frames = handFilm.frames;
    var n = frames.length;

    var normalised = [];
    for (var i = 0; i < n; i++) {
        var landmarks = frames[i].landmarks;
        var wrist = landmarks[0];

        // Step 1: handedness flip (mirror left hand across x axis).
        var xSign = frames[i].left_or_right === "left" ? -1 : 1;

        // Step 2: wrist-relative translation (with x flipped for left hands).
        var rel = [];
        for (var j = 0; j < LANDMARKS; j++) {
            rel.push([
                (landmarks[j].x - wrist.x) * xSign,
                landmarks[j].y - wrist.y,
                landmarks[j].z - wrist.z
            ]);
        }

        // Step 3: scale by wrist-to-landmark-9 distance (middle finger MCP).
        var lm9 = rel[9];
        var scale = Math.sqrt(lm9[0]*lm9[0] + lm9[1]*lm9[1] + lm9[2]*lm9[2]);
        if (scale < 1e-6) scale = 1.0;
        for (var j = 0; j < LANDMARKS; j++) {
            rel[j][0] /= scale;
            rel[j][1] /= scale;
            rel[j][2] /= scale;
        }

        // Step 4: rotation normalisation.
        //   up           = wrist → middle_MCP (rel[9], unit after scale step)
        //   rightApprox  = index_MCP → pinky_MCP (rel[17] − rel[5])
        //   right        = rightApprox orthogonalised against up, then normalised
        //   forward      = up × right
        // Projecting each landmark onto (right, up, forward) yields coords in a
        // canonical palm-aligned frame, independent of forearm orientation.
        var up = [rel[9][0], rel[9][1], rel[9][2]];
        var rApprox = [
            rel[17][0] - rel[5][0],
            rel[17][1] - rel[5][1],
            rel[17][2] - rel[5][2]
        ];
        var dotRU = rApprox[0]*up[0] + rApprox[1]*up[1] + rApprox[2]*up[2];
        var right = [
            rApprox[0] - dotRU * up[0],
            rApprox[1] - dotRU * up[1],
            rApprox[2] - dotRU * up[2]
        ];
        var rMag = Math.sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
        if (rMag < 1e-6) {
            // Degenerate palm (all knuckles collinear with wrist→middle_MCP);
            // fall back to world x axis so output stays finite.
            right = [1, 0, 0];
        } else {
            right[0] /= rMag; right[1] /= rMag; right[2] /= rMag;
        }
        var forward = [
            up[1]*right[2] - up[2]*right[1],
            up[2]*right[0] - up[0]*right[2],
            up[0]*right[1] - up[1]*right[0]
        ];

        for (var j = 0; j < LANDMARKS; j++) {
            var p = rel[j];
            rel[j] = [
                p[0]*right[0]   + p[1]*right[1]   + p[2]*right[2],
                p[0]*up[0]      + p[1]*up[1]      + p[2]*up[2],
                p[0]*forward[0] + p[1]*forward[1] + p[2]*forward[2]
            ];
        }

        normalised.push(rel);
    }

    // Velocity: frame-to-frame delta, zero for first frame
    var velocity = [];
    for (var i = 0; i < n; i++) {
        var v = [];
        if (i === 0) {
            for (var j = 0; j < LANDMARKS; j++) v.push([0, 0, 0]);
        } else {
            for (var j = 0; j < LANDMARKS; j++) {
                v.push([
                    normalised[i][j][0] - normalised[i-1][j][0],
                    normalised[i][j][1] - normalised[i-1][j][1],
                    normalised[i][j][2] - normalised[i-1][j][2]
                ]);
            }
        }
        velocity.push(v);
    }

    // Flatten each frame into 126 values: [coords(63), velocity(63)]
    var rows = [];
    for (var i = 0; i < n; i++) {
        var row = [];
        for (var j = 0; j < LANDMARKS; j++) {
            row.push(normalised[i][j][0], normalised[i][j][1], normalised[i][j][2]);
        }
        for (var j = 0; j < LANDMARKS; j++) {
            row.push(velocity[i][j][0], velocity[i][j][1], velocity[i][j][2]);
        }
        rows.push(row);
    }

    // Pad (append zeros) or trim (keep last TARGET_FRAMES) to exactly TARGET_FRAMES rows
    var zero = [];
    for (var k = 0; k < FEATURES_PER_FRAME; k++) zero.push(0);

    var result = [];
    if (n >= TARGET_FRAMES) {
        result = rows.slice(n - TARGET_FRAMES);
    } else {
        result = rows.slice();
        while (result.length < TARGET_FRAMES) result.push(zero.slice());
    }

    // Flatten to a single array
    var flat = [];
    for (var i = 0; i < TARGET_FRAMES; i++) {
        for (var k = 0; k < FEATURES_PER_FRAME; k++) flat.push(result[i][k]);
    }
    return flat;
}

/**
 * Same as featureMatrix but returns an array of TARGET_FRAMES rows,
 * each of length FEATURES_PER_FRAME (126).
 */
function featureRows(handFilm) {
    var flat = featureMatrix(handFilm);
    var rows = [];
    for (var i = 0; i < TARGET_FRAMES; i++) {
        rows.push(flat.slice(i * FEATURES_PER_FRAME, (i + 1) * FEATURES_PER_FRAME));
    }
    return rows;
}

/**
 * 256-element statistical summary vector used as MLP input.
 *
 *   63  — column-wise mean of normalised coords across frames
 *   63  — column-wise std  of normalised coords across frames
 *   63  — column-wise mean of velocity across frames
 *   63  — column-wise std  of velocity across frames
 *    3  — net raw wrist displacement (last − first frame, pre-normalisation)
 *    1  — dominant motion axis magnitude
 */
function summaryFeatures(handFilm) {
    var flat = featureMatrix(handFilm); // length TARGET_FRAMES * 126

    // Split into coords (cols 0–62) and velocities (cols 63–125) per frame
    var coords = []; // TARGET_FRAMES × 63
    var vels   = []; // TARGET_FRAMES × 63
    for (var i = 0; i < TARGET_FRAMES; i++) {
        var base = i * FEATURES_PER_FRAME;
        coords.push(flat.slice(base, base + COORDS_PER_FRAME));
        vels.push(flat.slice(base + COORDS_PER_FRAME, base + FEATURES_PER_FRAME));
    }

    // Column-wise mean
    function colMean(m) {
        var out = [];
        for (var j = 0; j < COORDS_PER_FRAME; j++) {
            var s = 0;
            for (var i = 0; i < TARGET_FRAMES; i++) s += m[i][j];
            out.push(s / TARGET_FRAMES);
        }
        return out;
    }

    // Column-wise population std
    function colStd(m, means) {
        var out = [];
        for (var j = 0; j < COORDS_PER_FRAME; j++) {
            var v = 0;
            for (var i = 0; i < TARGET_FRAMES; i++) {
                var d = m[i][j] - means[j];
                v += d * d;
            }
            out.push(Math.sqrt(v / TARGET_FRAMES));
        }
        return out;
    }

    var coordMean = colMean(coords);
    var coordStd  = colStd(coords, coordMean);
    var velMean   = colMean(vels);
    var velStd    = colStd(vels, velMean);

    // Net raw wrist displacement (first vs last frame, before normalisation).
    // X is flipped per-frame for left hands so left/right data aligns.
    var displacement = [0, 0, 0];
    var frames = handFilm.frames;
    if (frames.length >= 2) {
        var firstFrame = frames[0];
        var lastFrame  = frames[frames.length - 1];
        var firstSign = firstFrame.left_or_right === "left" ? -1 : 1;
        var lastSign  = lastFrame.left_or_right  === "left" ? -1 : 1;
        var first = firstFrame.landmarks[0];
        var last  = lastFrame.landmarks[0];
        displacement = [
            last.x * lastSign - first.x * firstSign,
            last.y - first.y,
            last.z - first.z
        ];
    }

    // Dominant motion axis: max |mean velocity| across x/y/z averaged over all landmarks
    var axisSum = [0, 0, 0];
    for (var i = 0; i < TARGET_FRAMES; i++) {
        for (var lm = 0; lm < LANDMARKS; lm++) {
            var b = lm * 3;
            axisSum[0] += vels[i][b];
            axisSum[1] += vels[i][b + 1];
            axisSum[2] += vels[i][b + 2];
        }
    }
    var total = TARGET_FRAMES * LANDMARKS;
    var dominant = Math.max(
        Math.abs(axisSum[0] / total),
        Math.abs(axisSum[1] / total),
        Math.abs(axisSum[2] / total)
    );

    return coordMean.concat(coordStd, velMean, velStd, displacement, [dominant]);
}
