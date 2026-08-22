"""A lightweight demo pipeline for sign-to-sentence prototyping.

The demo intentionally uses transparent template matching rather than claiming a
trained deep-learning model. Users can record a small vocabulary locally and the
classifier compares live landmark frames against those templates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from sign_language_translator.models.language_models import MixerLM, NgramLanguageModel

FEATURE_VERSION_LEGACY = 1
FEATURE_VERSION_ENHANCED = 2
POSITION_FEATURE_WEIGHT = float(os.getenv("SLT_POSITION_FEATURE_WEIGHT", "1.0"))
MIN_STABLE_PREDICTION_FRAMES = int(os.getenv("MIN_STABLE_PREDICTION_FRAMES", "2"))
MIN_ACCEPTED_CONFIDENCE = float(os.getenv("MIN_ACCEPTED_CONFIDENCE", "0.60"))
SHAPE_FEATURE_WEIGHT = float(os.getenv("SLT_SHAPE_FEATURE_WEIGHT", "2.25"))
DEPTH_FEATURE_WEIGHT = float(os.getenv("SLT_DEPTH_FEATURE_WEIGHT", "1.35"))
CONFIDENCE_MARGIN_THRESHOLD = float(os.getenv("SLT_CONFIDENCE_MARGIN_THRESHOLD", "0.03"))
CONFUSABLE_MARGIN_THRESHOLD = float(os.getenv("SLT_CONFUSABLE_MARGIN_THRESHOLD", "0.08"))

CONFUSABLE_LABEL_PAIRS = {
    frozenset({"gun", "point"}),
    frozenset({"thumbs_up", "spiderman"}),
}


def append_caption_word(history: Sequence[str], token: str, *, max_visible: int = 20) -> List[str]:
    """Append a recognized label only when it changes, trimming to the last visible block."""
    label = str(token or '').strip()
    if not label:
        return list(history)

    next_history = list(history)
    if next_history and next_history[-1] == label:
        return next_history

    next_history.append(label)
    if len(next_history) > max_visible:
        next_history = next_history[-max_visible:]
    return next_history


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.zeros_like(vector, dtype=float)
    return (vector / norm).astype(float)


def _coerce_21x3_landmarks(hand_landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    if not hand_landmarks:
        return np.zeros((21, 3), dtype=float)

    landmarks = np.asarray(hand_landmarks, dtype=float)
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(1, -1)

    if landmarks.size == 0:
        return np.zeros((21, 3), dtype=float)

    target_len = 42 if landmarks.shape[0] > 21 else 21
    if landmarks.shape[0] < target_len:
        landmarks = np.pad(landmarks, ((0, target_len - landmarks.shape[0]), (0, 0)))
    elif landmarks.shape[0] > target_len:
        landmarks = landmarks[:target_len]

    if landmarks.shape[1] < 3:
        landmarks = np.pad(landmarks, ((0, 0), (0, 3 - landmarks.shape[1])))

    return landmarks[:, :3].astype(float)


def canonicalize_landmarks(hand_landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Align landmarks to a canonical palm frame for rotation/tilt invariance."""

    landmarks = _coerce_21x3_landmarks(hand_landmarks)
    wrist = landmarks[0]

    middle_mcp = landmarks[9]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    y_axis = _safe_normalize(middle_mcp - wrist)
    x_axis = _safe_normalize(index_mcp - pinky_mcp)
    z_axis = _safe_normalize(np.cross(x_axis, y_axis))

    x_axis = _safe_normalize(np.cross(y_axis, z_axis))
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    if np.allclose(x_axis, 0.0) or np.allclose(y_axis, 0.0) or np.allclose(z_axis, 0.0):
        return landmarks - wrist

    centered = landmarks - wrist
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return centered @ rotation


def normalize_landmarks(hand_landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Canonicalize, center, and scale landmarks for invariant position features."""

    canonical = canonicalize_landmarks(hand_landmarks)
    centered = canonical - canonical[0]
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(float)


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba_raw = a - b
    bc_raw = c - b
    if float(np.linalg.norm(ba_raw)) < 1e-8 or float(np.linalg.norm(bc_raw)) < 1e-8:
        return 0.0

    ba = _safe_normalize(ba_raw)
    bc = _safe_normalize(bc_raw)
    cosine = float(np.clip(np.dot(ba, bc), -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)


def _finger_shape_features(normalized: np.ndarray) -> np.ndarray:
    curl_triplets = [
        ((0, 1, 2), (1, 2, 3), (2, 3, 4)),
        ((0, 5, 6), (5, 6, 7), (6, 7, 8)),
        ((0, 9, 10), (9, 10, 11), (10, 11, 12)),
        ((0, 13, 14), (13, 14, 15), (14, 15, 16)),
        ((0, 17, 18), (17, 18, 19), (18, 19, 20)),
    ]
    curls: List[float] = []
    for group in curl_triplets:
        for a, b, c in group:
            curls.append(_joint_angle(normalized[a], normalized[b], normalized[c]))

    spread_pairs = [(5, 9), (9, 13), (13, 17), (5, 17)]
    spreads = [_joint_angle(normalized[left], normalized[0], normalized[right]) for left, right in spread_pairs]

    features = curls + spreads

    if normalized.shape[0] >= 42:
        offset = 21
        for group in curl_triplets:
            for a, b, c in group:
                features.append(_joint_angle(normalized[a + offset], normalized[b + offset], normalized[c + offset]))
        for left, right in spread_pairs:
            features.append(_joint_angle(normalized[left + offset], normalized[offset], normalized[right + offset]))

    return np.asarray(features, dtype=float)


def _build_legacy_landmark_vector(hand_landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    if not hand_landmarks:
        return np.zeros(0, dtype=float)

    landmarks = np.asarray(hand_landmarks, dtype=float)
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(1, -1)
    if landmarks.shape[0] <= 3:
        if landmarks.shape[0] == 0:
            return np.zeros(9, dtype=float)
        landmarks = landmarks[:3]
        if landmarks.shape[0] < 3:
            landmarks = np.pad(landmarks, ((0, 3 - landmarks.shape[0]), (0, 0)))

        mid_index = max(1, min(2, landmarks.shape[0] // 2))
        features = [
            landmarks[0, 0],
            landmarks[0, 1],
            landmarks[-1, 1],
            landmarks[0, 2] if landmarks.shape[1] > 2 else 0.0,
            landmarks[mid_index, 0] if landmarks.shape[0] > mid_index else 0.0,
            landmarks[mid_index, 1] if landmarks.shape[0] > mid_index else 0.0,
            landmarks[-1, 0],
            landmarks[-1, 1],
            landmarks[-1, 2] if landmarks.shape[1] > 2 else 0.0,
        ]
        return np.asarray(features, dtype=float)

    return normalize_landmarks(hand_landmarks).reshape(-1).astype(float)


def build_landmark_vector(
    hand_landmarks: Sequence[Sequence[float]], *, feature_version: int = FEATURE_VERSION_ENHANCED
) -> np.ndarray:
    """Flatten hand landmarks into a versioned feature vector."""

    if feature_version == FEATURE_VERSION_LEGACY:
        return _build_legacy_landmark_vector(hand_landmarks)

    if not hand_landmarks:
        return np.zeros(0, dtype=float)

    landmarks = np.asarray(hand_landmarks, dtype=float)
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(1, -1)
    if landmarks.shape[0] <= 3:
        return _build_legacy_landmark_vector(hand_landmarks)

    normalized = normalize_landmarks(hand_landmarks)
    if float(np.linalg.norm(normalized, axis=1).max()) < 1e-6:
        return np.zeros(82, dtype=float)

    position_features = normalized.reshape(-1).astype(float)
    shape_features = _finger_shape_features(normalized)
    return np.concatenate([position_features, shape_features]).astype(float)


@dataclass
class GestureClassifier:
    """Template-matching gesture classifier for a user-recorded vocabulary."""

    template_store_path: str | Path = field(
        default_factory=lambda: os.getenv("SLT_DEMO_GESTURE_STORE", ".demo_gestures.json")
    )
    low_confidence_threshold: float = 0.60
    ambiguity_ratio_threshold: float = 0.05
    templates: Dict[str, np.ndarray] = field(default_factory=dict)
    template_feature_versions: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.template_store_path = Path(self.template_store_path)
        self.templates = {}
        self.template_feature_versions = {}
        self._load_from_disk()
        if not self.templates:
            self._seed_default_templates()

    def _seed_default_templates(self) -> None:
        """The live demo intentionally avoids shipping fake landmark templates.

        Users record their own gesture vocabulary in the guided setup flow instead of
        fabricating hand templates that were never actually captured.
        """
        return

    def _storage_path(self) -> Path:
        self.template_store_path.parent.mkdir(parents=True, exist_ok=True)
        return self.template_store_path

    def _load_from_disk(self) -> None:
        path = self._storage_path()
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(payload, dict):
            return

        if "templates" in payload and isinstance(payload["templates"], dict):
            for label, item in payload["templates"].items():
                if isinstance(item, dict) and isinstance(item.get("vector"), list):
                    self.templates[str(label)] = np.asarray(item["vector"], dtype=float)
                    self.template_feature_versions[str(label)] = int(
                        item.get("feature_version", FEATURE_VERSION_ENHANCED)
                    )
            return

        # Backward compatibility with v1 store: {label: [vector]}.
        for label, vector in payload.items():
            if isinstance(vector, list):
                self.templates[str(label)] = np.asarray(vector, dtype=float)
                self.template_feature_versions[str(label)] = FEATURE_VERSION_LEGACY

    def _persist(self) -> None:
        payload = {
            "_meta": {"schema_version": 2, "current_feature_version": FEATURE_VERSION_ENHANCED},
            "templates": {
                label: {
                    "feature_version": int(
                        self.template_feature_versions.get(label, FEATURE_VERSION_ENHANCED)
                    ),
                    "vector": vector.tolist(),
                }
                for label, vector in self.templates.items()
            },
        }
        with self._storage_path().open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _coerce_label(self, label: str) -> str:
        sanitized = str(label).strip().lower()
        if not sanitized:
            raise ValueError("Gesture label is required.")
        return sanitized

    def record_template(self, label: str, samples: Sequence[Sequence[Sequence[float]]], *, store: bool = True) -> np.ndarray:
        sanitized = self._coerce_label(label)
        if len(self.templates) >= 50 and sanitized not in self.templates:
            raise ValueError("A maximum of 50 templates is supported.")

        vectors = [
            build_landmark_vector(sample, feature_version=FEATURE_VERSION_ENHANCED)
            for sample in samples
            if isinstance(sample, Sequence) and len(sample) > 0
        ]
        if not vectors:
            raise ValueError("At least one valid sample is required.")

        template = np.mean(np.vstack(vectors), axis=0)
        self.templates[sanitized] = template
        self.template_feature_versions[sanitized] = FEATURE_VERSION_ENHANCED
        if store:
            self._persist()
        return template

    def delete_template(self, label: str) -> bool:
        sanitized = self._coerce_label(label)
        if sanitized in self.templates:
            del self.templates[sanitized]
            self.template_feature_versions.pop(sanitized, None)
            self._persist()
            return True
        return False

    def template_names(self) -> List[str]:
        return sorted(self.templates.keys())

    def template_migration_status(self) -> Dict[str, bool]:
        return {
            label: self.template_feature_versions.get(label, FEATURE_VERSION_LEGACY)
            < FEATURE_VERSION_ENHANCED
            for label in self.template_names()
        }

    def _mirrored_vector(self, window: Sequence[Sequence[float]]) -> np.ndarray:
        landmarks = np.asarray(window, dtype=float)
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        if landmarks.size == 0:
            return np.zeros(0, dtype=float)

        mirrored = landmarks.copy()
        mirrored[:, 0] = -mirrored[:, 0]
        return np.asarray(
            build_landmark_vector(mirrored.tolist(), feature_version=FEATURE_VERSION_ENHANCED),
            dtype=float,
        ).reshape(-1)

    def _confidence_from_distance(self, distance: float, feature_length: int) -> float:
        return max(0.0, min(1.0, np.exp(-distance / max(1.0, feature_length / 14.0))))

    def _weighted_distance(self, source: np.ndarray, template: np.ndarray, feature_version: int) -> float:
        if feature_version < FEATURE_VERSION_ENHANCED:
            length = max(len(source), len(template))
            padded_source = np.pad(source, (0, max(0, length - len(source))))
            padded_template = np.pad(template, (0, max(0, length - len(template))))
            return float(np.linalg.norm(padded_source - padded_template))

        length = max(len(source), len(template))
        padded_source = np.pad(source, (0, max(0, length - len(source))))
        padded_template = np.pad(template, (0, max(0, length - len(template))))

        pos_len = 126 if length > 82 else 63
        
        source_pos = padded_source[:pos_len].reshape(-1, 3)
        template_pos = padded_template[:pos_len].reshape(-1, 3)
        pos_delta = source_pos - template_pos
        pos_delta[:, 2] *= DEPTH_FEATURE_WEIGHT
        pos_norm = float(np.linalg.norm(pos_delta.reshape(-1)))

        source_shape = padded_source[pos_len:]
        template_shape = padded_template[pos_len:]
        shape_norm = float(np.linalg.norm(source_shape - template_shape))

        return float(
            np.sqrt((POSITION_FEATURE_WEIGHT * pos_norm) ** 2 + (SHAPE_FEATURE_WEIGHT * shape_norm) ** 2)
        )

    def predict_with_diagnostics(self, landmark_windows: Iterable[Sequence[Sequence[float]]]) -> Dict[str, Any]:
        for window in landmark_windows:
            enhanced_vector = np.asarray(
                build_landmark_vector(window, feature_version=FEATURE_VERSION_ENHANCED), dtype=float
            ).reshape(-1)
            legacy_vector = np.asarray(
                build_landmark_vector(window, feature_version=FEATURE_VERSION_LEGACY), dtype=float
            ).reshape(-1)
            if enhanced_vector.size == 0 or np.allclose(enhanced_vector, 0.0):
                return {
                    "token": "unknown",
                    "confidence": 0.0,
                    "reason": "empty_or_degenerate_features",
                    "top_candidates": [],
                    "ambiguous": True,
                    "confidence_margin": 0.0,
                    "confusable_pair": False,
                }

            mirrored_vector = self._mirrored_vector(window)
            if not self.templates:
                return {
                    "token": "unknown",
                    "confidence": 0.0,
                    "reason": "no_templates",
                    "top_candidates": [],
                    "ambiguous": True,
                    "confidence_margin": 0.0,
                    "confusable_pair": False,
                }

            scored: List[Tuple[str, float, float]] = []
            for label, template in self.templates.items():
                template_vector = np.asarray(template, dtype=float).reshape(-1)
                version = self.template_feature_versions.get(label, FEATURE_VERSION_LEGACY)
                source_vector = enhanced_vector if version >= FEATURE_VERSION_ENHANCED else legacy_vector
                base_distance = self._weighted_distance(source_vector, template_vector, version)

                if version >= FEATURE_VERSION_ENHANCED and mirrored_vector.size > 0:
                    mirrored_distance = self._weighted_distance(mirrored_vector, template_vector, version)
                    best_distance = min(base_distance, mirrored_distance)
                else:
                    best_distance = base_distance

                feature_length = max(len(source_vector), len(template_vector))
                candidate_confidence = self._confidence_from_distance(best_distance, feature_length)
                scored.append((label, best_distance, candidate_confidence))

            scored.sort(key=lambda item: item[1])
            top_candidates = [
                {
                    "label": label,
                    "distance": round(float(distance), 5),
                    "confidence": round(float(candidate_confidence), 5),
                }
                for label, distance, candidate_confidence in scored[:2]
            ]

            best_label, best_distance, best_confidence = scored[0]
            if len(scored) > 1:
                second_label, second_distance, second_confidence = scored[1]
                ambiguity_ratio = (second_distance - best_distance) / max(second_distance, 1e-8)
                confidence_margin = best_confidence - second_confidence
                confusable_pair = frozenset({best_label, second_label}) in CONFUSABLE_LABEL_PAIRS
                margin_threshold = CONFUSABLE_MARGIN_THRESHOLD if confusable_pair else CONFIDENCE_MARGIN_THRESHOLD
            else:
                ambiguity_ratio = float("inf")
                confidence_margin = 1.0
                confusable_pair = False
                margin_threshold = CONFIDENCE_MARGIN_THRESHOLD

            ambiguous = False
            reason = "matched"
            if np.isfinite(ambiguity_ratio) and ambiguity_ratio < self.ambiguity_ratio_threshold:
                ambiguous = True
                reason = "distance_ambiguity"
            if confidence_margin < margin_threshold:
                ambiguous = True
                reason = "confidence_margin"

            if best_confidence < self.low_confidence_threshold:
                return {
                    "token": "unknown",
                    "confidence": float(best_confidence),
                    "reason": "low_confidence",
                    "top_candidates": top_candidates,
                    "ambiguous": True,
                    "confidence_margin": float(confidence_margin),
                    "confusable_pair": bool(confusable_pair),
                }
            if ambiguous:
                return {
                    "token": "unknown",
                    "confidence": float(best_confidence),
                    "reason": reason,
                    "top_candidates": top_candidates,
                    "ambiguous": True,
                    "confidence_margin": float(confidence_margin),
                    "confusable_pair": bool(confusable_pair),
                }

            return {
                "token": best_label,
                "confidence": float(best_confidence),
                "reason": "matched",
                "top_candidates": top_candidates,
                "ambiguous": False,
                "confidence_margin": float(confidence_margin),
                "confusable_pair": bool(confusable_pair),
            }

        return {
            "token": "unknown",
            "confidence": 0.0,
            "reason": "no_windows",
            "top_candidates": [],
            "ambiguous": True,
            "confidence_margin": 0.0,
            "confusable_pair": False,
        }

    def predict(self, landmark_windows: Iterable[Sequence[Sequence[float]]]) -> Tuple[str, float]:
        details = self.predict_with_diagnostics(landmark_windows)
        return str(details.get("token", "unknown")), float(details.get("confidence", 0.0))


class DemoPipeline:
    """Combine a lightweight classifier with a small language model."""

    def __init__(self, gesture_store_path: str | Path | None = None) -> None:
        self.classifier = GestureClassifier(
            template_store_path=gesture_store_path or os.getenv("SLT_DEMO_GESTURE_STORE", ".demo_gestures.json")
        )
        self.language_model = NgramLanguageModel(window_size=2, name="demo-sentence")
        self.language_model.fit([
            ["open_hand", "hello"],
            ["open_hand", "thank_you"],
            ["point", "help"],
            ["thumbs_up", "good"],
        ])

        self.mixer = MixerLM(
            models=[self.language_model],
            selection_probabilities=[1.0],
            name="demo-mixer",
            model_selection_strategy="choose",
        )
        self.token_history: List[str] = []

    def process_gloss(self, gloss: str) -> str:
        self.token_history.append(gloss)
        if len(self.token_history) < 2:
            return gloss

        token, _ = self.mixer.next(self.token_history)
        return " ".join(self.token_history + [str(token)])


__all__ = [
    "DemoPipeline",
    "GestureClassifier",
    "build_landmark_vector",
    "normalize_landmarks",
]
