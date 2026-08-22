/**
 * matcher.js — Client-side gesture template matching.
 *
 * Behaviorally equivalent port of sign_language_translator/demo/pipeline.py.
 * All constants, formulas, and thresholds match the Python source exactly so
 * accuracy does not regress when we remove the WebSocket classification path.
 */

// ─── Constants (mirrors pipeline.py) ────────────────────────────────────────
const POSITION_FEATURE_WEIGHT = 1.0;
const SHAPE_FEATURE_WEIGHT = 2.25;
const DEPTH_FEATURE_WEIGHT = 1.35;
const MIN_ACCEPTED_CONFIDENCE = 0.60;
const CONFIDENCE_MARGIN_THRESHOLD = 0.03;
const CONFUSABLE_MARGIN_THRESHOLD = 0.08;
const MIN_STABLE_PREDICTION_FRAMES = 2;

const CONFUSABLE_PAIRS = [
  new Set(['gun', 'point']),
  new Set(['thumbs_up', 'spiderman']),
];

function isConfusablePair(a, b) {
  return CONFUSABLE_PAIRS.some(pair => pair.has(a) && pair.has(b));
}

// ─── Vector math helpers ─────────────────────────────────────────────────────
function norm(v) {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function safeNormalize(v) {
  const n = norm(v);
  if (!isFinite(n) || n < 1e-8) return v.map(() => 0);
  return v.map(x => x / n);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale3(v, s) { return [v[0] * s, v[1] * s, v[2] * s]; }

// ─── Landmark coercion ───────────────────────────────────────────────────────
/**
 * Accept landmarks as an array of {x,y,z} objects OR [x,y,z] arrays.
 * Returns a flat Float64Array shaped (N, 3) stored row-major.
 */
function coerce21x3(handLandmarks) {
  if (!handLandmarks || handLandmarks.length === 0) {
    return new Float64Array(21 * 3);
  }
  const out = new Float64Array(21 * 3);
  const n = Math.min(handLandmarks.length, 21);
  for (let i = 0; i < n; i++) {
    const p = handLandmarks[i];
    if (Array.isArray(p)) {
      out[i * 3 + 0] = p[0] || 0;
      out[i * 3 + 1] = p[1] || 0;
      out[i * 3 + 2] = p[2] || 0;
    } else {
      out[i * 3 + 0] = p.x || 0;
      out[i * 3 + 1] = p.y || 0;
      out[i * 3 + 2] = p.z || 0;
    }
  }
  return out;
}

function getLandmark(flat, i) {
  return [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]];
}

// ─── Normalization (mirrors canonicalize_landmarks + normalize_landmarks) ───
export function canonicalizeLandmarks(handLandmarks) {
  const lm = coerce21x3(handLandmarks);
  const wrist = getLandmark(lm, 0);
  const middleMcp = getLandmark(lm, 9);
  const indexMcp = getLandmark(lm, 5);
  const pinkyMcp = getLandmark(lm, 17);

  let yAxis = safeNormalize(sub3(middleMcp, wrist));
  let xAxis = safeNormalize(sub3(indexMcp, pinkyMcp));
  let zAxis = safeNormalize(cross3(xAxis, yAxis));
  xAxis = safeNormalize(cross3(yAxis, zAxis));
  yAxis = safeNormalize(cross3(zAxis, xAxis));

  const degenerate =
    norm(xAxis) < 0.5 || norm(yAxis) < 0.5 || norm(zAxis) < 0.5;

  const out = new Float64Array(21 * 3);
  for (let i = 0; i < 21; i++) {
    const p = getLandmark(lm, i);
    const c = sub3(p, wrist);
    if (degenerate) {
      out[i * 3 + 0] = c[0];
      out[i * 3 + 1] = c[1];
      out[i * 3 + 2] = c[2];
    } else {
      // rotate: out = c @ [xAxis, yAxis, zAxis]  (same as Python `centered @ rotation`)
      out[i * 3 + 0] = dot(c, xAxis);
      out[i * 3 + 1] = dot(c, yAxis);
      out[i * 3 + 2] = dot(c, zAxis);
    }
  }
  return out; // flat Float64Array, 21×3 row-major
}

export function normalizeLandmarks(handLandmarks) {
  const canonical = canonicalizeLandmarks(handLandmarks);
  // Re-center on wrist (index 0) — already at origin after canonicalize
  let maxNorm = 0;
  for (let i = 0; i < 21; i++) {
    const x = canonical[i * 3], y = canonical[i * 3 + 1], z = canonical[i * 3 + 2];
    const n = Math.sqrt(x * x + y * y + z * z);
    if (n > maxNorm) maxNorm = n;
  }
  if (!isFinite(maxNorm) || maxNorm < 1e-8) maxNorm = 1.0;
  const out = new Float64Array(21 * 3);
  for (let i = 0; i < 63; i++) out[i] = canonical[i] / maxNorm;
  return out;
}

// ─── Feature extraction (mirrors _finger_shape_features) ────────────────────
function jointAngle(norm21x3, a, b, c) {
  const pa = getLandmark(norm21x3, a);
  const pb = getLandmark(norm21x3, b);
  const pc = getLandmark(norm21x3, c);
  const ba = sub3(pa, pb);
  const bc = sub3(pc, pb);
  const nba = norm(ba), nbc = norm(bc);
  if (nba < 1e-8 || nbc < 1e-8) return 0;
  const cosine = Math.max(-1, Math.min(1, dot(ba, bc) / (nba * nbc)));
  return Math.acos(cosine) / Math.PI;
}

const CURL_TRIPLETS = [
  [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
  [[0, 5, 6], [5, 6, 7], [6, 7, 8]],
  [[0, 9, 10], [9, 10, 11], [10, 11, 12]],
  [[0, 13, 14], [13, 14, 15], [14, 15, 16]],
  [[0, 17, 18], [17, 18, 19], [18, 19, 20]],
];
const SPREAD_PAIRS = [[5, 9], [9, 13], [13, 17], [5, 17]];

function fingerShapeFeatures(norm21x3) {
  const features = [];
  for (const group of CURL_TRIPLETS) {
    for (const [a, b, c] of group) {
      features.push(jointAngle(norm21x3, a, b, c));
    }
  }
  for (const [l, r] of SPREAD_PAIRS) {
    features.push(jointAngle(norm21x3, l, 0, r));
  }
  return features;
}

// ─── Build feature vector (mirrors build_landmark_vector ENHANCED) ───────────
export function buildLandmarkVector(handLandmarks) {
  if (!handLandmarks || handLandmarks.length < 4) return new Float64Array(0);
  const norm21 = normalizeLandmarks(handLandmarks);
  // Check degenerate
  let maxV = 0;
  for (const v of norm21) if (Math.abs(v) > maxV) maxV = Math.abs(v);
  if (maxV < 1e-6) return new Float64Array(82); // all-zero sentinel

  const posFeatures = Array.from(norm21); // 63 values
  const shapeFeatures = fingerShapeFeatures(norm21); // 19 values → total 82
  const out = new Float64Array(posFeatures.length + shapeFeatures.length);
  out.set(posFeatures, 0);
  out.set(shapeFeatures, posFeatures.length);
  return out;
}

// Build mirrored version (flip X axis, mirrors _mirrored_vector)
function buildMirroredVector(handLandmarks) {
  if (!handLandmarks || handLandmarks.length === 0) return new Float64Array(0);
  const mirrored = handLandmarks.map(p => {
    if (Array.isArray(p)) return [-p[0], p[1], p[2]];
    return { x: -p.x, y: p.y, z: p.z };
  });
  return buildLandmarkVector(mirrored);
}

// ─── Distance & confidence (mirrors _weighted_distance + _confidence_from_distance)
function weightedDistance(source, template) {
  const length = Math.max(source.length, template.length);
  const src = length > source.length
    ? (() => { const a = new Float64Array(length); a.set(source); return a; })()
    : source;
  const tpl = length > template.length
    ? (() => { const a = new Float64Array(length); a.set(template); return a; })()
    : template;

  const posLen = length > 82 ? 126 : 63;

  // Position distance with depth weighting
  let posSq = 0;
  for (let i = 0; i < posLen; i++) {
    let d = src[i] - tpl[i];
    if ((i % 3) === 2) d *= DEPTH_FEATURE_WEIGHT; // z component
    posSq += d * d;
  }
  const posNorm = Math.sqrt(posSq);

  // Shape distance
  let shapeSq = 0;
  for (let i = posLen; i < length; i++) {
    const d = src[i] - tpl[i];
    shapeSq += d * d;
  }
  const shapeNorm = Math.sqrt(shapeSq);

  return Math.sqrt(
    (POSITION_FEATURE_WEIGHT * posNorm) ** 2 +
    (SHAPE_FEATURE_WEIGHT * shapeNorm) ** 2
  );
}

function confidenceFromDistance(distance, featureLength) {
  return Math.max(0, Math.min(1, Math.exp(-distance / Math.max(1, featureLength / 14))));
}

// ─── GestureMatcher class ────────────────────────────────────────────────────
export class GestureMatcher {
  constructor() {
    /** @type {Map<string, Float64Array>} label → averaged feature vector */
    this.templates = new Map();
    this._stableLabel = null;
    this._stableCount = 0;
    this.loadFromLocalStorage();
  }

  // ── Persistence ────────────────────────────────────────────────────────────
  loadFromLocalStorage() {
    try {
      const raw = localStorage.getItem('slt_gestures');
      if (!raw) return;
      const store = JSON.parse(raw);
      for (const [label, samples] of Object.entries(store)) {
        this._buildAndStoreTemplate(label, samples);
      }
    } catch (e) {
      console.warn('[matcher] Failed to load gestures from localStorage:', e);
    }
  }

  /**
   * Record a new template from raw landmark samples and persist to localStorage.
   * @param {string} label
   * @param {Array} samples  array of 21-point landmark arrays
   */
  recordTemplate(label, samples) {
    if (!label || !samples || samples.length === 0) return;
    const sanitized = String(label).trim().toLowerCase();

    // Build average vector from all samples
    this._buildAndStoreTemplate(sanitized, samples);

    // Persist
    try {
      const store = JSON.parse(localStorage.getItem('slt_gestures') || '{}');
      store[sanitized] = samples;
      localStorage.setItem('slt_gestures', JSON.stringify(store));
    } catch (e) {
      console.warn('[matcher] Failed to persist template:', e);
    }
  }

  deleteTemplate(label) {
    const sanitized = String(label).trim().toLowerCase();
    this.templates.delete(sanitized);
    try {
      const store = JSON.parse(localStorage.getItem('slt_gestures') || '{}');
      delete store[sanitized];
      localStorage.setItem('slt_gestures', JSON.stringify(store));
    } catch (e) {
      console.warn('[matcher] Failed to delete template:', e);
    }
  }

  getLabels() {
    return Array.from(this.templates.keys()).sort();
  }

  _buildAndStoreTemplate(label, samples) {
    const vectors = samples
      .map(s => buildLandmarkVector(s))
      .filter(v => v.length > 0);
    if (vectors.length === 0) return;

    // Average all sample vectors (mirrors np.mean(np.vstack(vectors), axis=0))
    const len = vectors[0].length;
    const mean = new Float64Array(len);
    for (const v of vectors) {
      for (let i = 0; i < len; i++) mean[i] += v[i];
    }
    for (let i = 0; i < len; i++) mean[i] /= vectors.length;
    this.templates.set(label, mean);
  }

  // ── Prediction ─────────────────────────────────────────────────────────────
  /**
   * Classify a single hand frame.
   * @param {Array} handLandmarks  21-point array of {x,y,z} or [x,y,z]
   * @returns {{ token: string, confidence: number, reason: string, top_candidates: Array, ambiguous: boolean, confidence_margin: number }}
   */
  predict(handLandmarks) {
    const result = this._predictWithDiagnostics(handLandmarks);
    // Temporal stabilization (mirrors MIN_STABLE_PREDICTION_FRAMES)
    const raw = result.token;
    if (raw === 'unknown' || result.ambiguous) {
      this._stableCount = 0;
      this._stableLabel = null;
      return { ...result, token: 'unknown' };
    }
    if (raw === this._stableLabel) {
      this._stableCount += 1;
    } else {
      this._stableLabel = raw;
      this._stableCount = 1;
    }
    if (this._stableCount < MIN_STABLE_PREDICTION_FRAMES) {
      return { ...result, token: 'unknown' };
    }
    return result;
  }

  _predictWithDiagnostics(handLandmarks) {
    const EMPTY = (reason) => ({
      token: 'unknown', confidence: 0, reason,
      top_candidates: [], ambiguous: true, confidence_margin: 0,
    });

    if (this.templates.size === 0) return EMPTY('no_templates');

    const enhancedVec = buildLandmarkVector(handLandmarks);
    if (enhancedVec.length === 0 || enhancedVec.every(v => v === 0)) {
      return EMPTY('empty_or_degenerate_features');
    }
    const mirroredVec = buildMirroredVector(handLandmarks);

    // Score every template
    const scored = [];
    for (const [label, templateVec] of this.templates) {
      const baseDist = weightedDistance(enhancedVec, templateVec);
      let bestDist = baseDist;
      if (mirroredVec.length > 0) {
        const mirDist = weightedDistance(mirroredVec, templateVec);
        if (mirDist < bestDist) bestDist = mirDist;
      }
      const featureLen = Math.max(enhancedVec.length, templateVec.length);
      const conf = confidenceFromDistance(bestDist, featureLen);
      scored.push({ label, distance: bestDist, confidence: conf });
    }
    scored.sort((a, b) => a.distance - b.distance);

    const top2 = scored.slice(0, 2).map(s => ({
      label: s.label,
      distance: Math.round(s.distance * 1e5) / 1e5,
      confidence: Math.round(s.confidence * 1e5) / 1e5,
    }));

    const best = scored[0];
    let confidenceMargin = 1.0;
    let ambiguous = false;
    let reason = 'matched';
    let confusablePair = false;

    if (scored.length > 1) {
      const second = scored[1];
      const ambiguityRatio = (second.distance - best.distance) / Math.max(second.distance, 1e-8);
      confidenceMargin = best.confidence - second.confidence;
      confusablePair = isConfusablePair(best.label, second.label);
      const marginThreshold = confusablePair ? CONFUSABLE_MARGIN_THRESHOLD : CONFIDENCE_MARGIN_THRESHOLD;

      if (isFinite(ambiguityRatio) && ambiguityRatio < 0.05) {
        ambiguous = true; reason = 'distance_ambiguity';
      }
      if (confidenceMargin < marginThreshold) {
        ambiguous = true; reason = 'confidence_margin';
      }
    }

    if (best.confidence < MIN_ACCEPTED_CONFIDENCE) {
      return {
        token: 'unknown', confidence: best.confidence, reason: 'low_confidence',
        top_candidates: top2, ambiguous: true, confidence_margin: confidenceMargin,
      };
    }
    if (ambiguous) {
      return {
        token: 'unknown', confidence: best.confidence, reason,
        top_candidates: top2, ambiguous: true, confidence_margin: confidenceMargin,
      };
    }

    return {
      token: best.label, confidence: best.confidence, reason: 'matched',
      top_candidates: top2, ambiguous: false, confidence_margin: confidenceMargin,
    };
  }
}
