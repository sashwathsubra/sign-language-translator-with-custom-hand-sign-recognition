import json
import numpy as np
import pytest

from sign_language_translator.demo.pipeline import (
    DemoPipeline,
    GestureClassifier,
    append_caption_word,
    build_landmark_vector,
    normalize_landmarks,
)


def make_landmark_sample(value=1.0):
    points = []
    for i in range(21):
        points.append([value * (i % 5) / 20.0, value * (i % 3) / 20.0, value * 0.1])
    return points


def make_pose_point_like():
    points = [[0.0, 0.0, 0.0] for _ in range(21)]
    points[0] = [0.0, 0.0, 0.0]
    points[5], points[6], points[7], points[8] = [0.15, 0.08, 0.0], [0.16, 0.22, 0.0], [0.17, 0.35, 0.0], [0.17, 0.48, 0.0]
    points[9], points[10], points[11], points[12] = [0.0, 0.08, 0.0], [0.02, 0.12, 0.0], [0.03, 0.14, 0.0], [0.04, 0.16, 0.0]
    points[13], points[14], points[15], points[16] = [-0.1, 0.07, 0.0], [-0.08, 0.11, 0.0], [-0.07, 0.13, 0.0], [-0.06, 0.15, 0.0]
    points[17], points[18], points[19], points[20] = [-0.2, 0.06, 0.0], [-0.17, 0.1, 0.0], [-0.16, 0.12, 0.0], [-0.15, 0.14, 0.0]
    points[1], points[2], points[3], points[4] = [0.08, -0.01, 0.0], [0.06, 0.02, 0.0], [0.04, 0.03, 0.0], [0.03, 0.02, 0.0]
    return points


def make_pose_gun_like():
    points = make_pose_point_like()
    points[1], points[2], points[3], points[4] = [0.08, 0.03, 0.0], [0.16, 0.08, 0.0], [0.24, 0.12, 0.0], [0.33, 0.14, 0.0]
    return points


def test_build_landmark_vector_flattens_hand_landmarks():
    hand_landmarks = [
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.0],
        [0.2, 0.4, 0.0],
    ]

    vector = build_landmark_vector(hand_landmarks)

    assert vector.shape == (9,)
    assert np.isclose(vector[0], 0.0)
    assert np.isclose(vector[2], 0.4)


def test_normalize_landmarks_is_translation_and_scale_invariant():
    sample_a = make_landmark_sample(1.0)
    sample_b = [[point[0] + 0.25, point[1] + 0.5, point[2]] for point in sample_a]
    sample_c = [[point[0] * 2.0, point[1] * 2.0, point[2]] for point in sample_a]

    norm_a = normalize_landmarks(sample_a)
    norm_b = normalize_landmarks(sample_b)
    norm_c = normalize_landmarks(sample_c)

    assert norm_a.shape == (21, 3)
    assert np.allclose(norm_a, norm_b, atol=1e-6)
    assert np.allclose(norm_a, norm_c, atol=1e-6)


def test_demo_pipeline_classifies_and_updates_sentence(tmp_path):
    pipeline = DemoPipeline(gesture_store_path=tmp_path / "demo_pipeline_templates.json")
    hand_landmarks = make_landmark_sample(0.4)
    pipeline.classifier.record_template("open_hand", [hand_landmarks, hand_landmarks])

    token, confidence = pipeline.classifier.predict([hand_landmarks])

    assert token == "open_hand"
    assert 0.0 <= confidence <= 1.0

    sentence = pipeline.process_gloss(token)
    assert isinstance(sentence, str)
    assert sentence


def test_append_caption_word_deduplicates_consecutive_labels():
    history = []
    history = append_caption_word(history, "hello")
    history = append_caption_word(history, "hello")
    history = append_caption_word(history, "hello")
    history = append_caption_word(history, "yes")

    assert history == ["hello", "yes"]


def test_append_caption_word_keeps_last_20_entries_without_touching_template_store():
    classifier = GestureClassifier()
    original_templates = list(classifier.template_names())

    history = [f"word_{index}" for index in range(25)]
    history = append_caption_word(history, "word_25", max_visible=20)

    assert len(history) == 20
    assert history[0] == "word_6"
    assert classifier.template_names() == original_templates


def test_template_classifier_records_and_matches_with_confidence_threshold(tmp_path):
    classifier = GestureClassifier(
        template_store_path=tmp_path / "threshold_templates.json",
        low_confidence_threshold=0.7,
    )
    sample = make_landmark_sample(0.7)
    classifier.record_template("peace", [sample, sample, sample])

    matched_label, confidence = classifier.predict([sample])
    assert matched_label == "peace"
    assert confidence >= 0.7

    low_quality = [[0.0, 0.0, 0.0] for _ in range(21)]
    unknown_label, unknown_conf = classifier.predict([low_quality])
    assert unknown_label == "unknown"
    assert unknown_conf < 0.7


def test_template_classifier_matches_mirrored_hand_sample(tmp_path):
    classifier = GestureClassifier(
        template_store_path=tmp_path / "mirror_templates.json",
        low_confidence_threshold=0.6,
    )
    sample = make_landmark_sample(0.9)
    mirrored_sample = [[-point[0], point[1], point[2]] for point in sample]
    classifier.record_template("gun", [sample, sample])

    matched_label, confidence = classifier.predict([mirrored_sample])
    assert matched_label == "gun"
    assert confidence >= 0.6


def test_similar_synthetic_poses_do_not_cross_match(tmp_path):
    classifier = GestureClassifier(
        template_store_path=tmp_path / "distinct_poses.json",
        low_confidence_threshold=0.45,
    )
    point_pose = make_pose_point_like()
    gun_pose = make_pose_gun_like()

    classifier.record_template("point", [point_pose, point_pose])
    classifier.record_template("gun", [gun_pose, gun_pose])

    point_label, point_conf = classifier.predict([point_pose])
    gun_label, gun_conf = classifier.predict([gun_pose])

    assert point_label == "point"
    assert gun_label == "gun"
    assert point_conf >= 0.45
    assert gun_conf >= 0.45


def test_malformed_landmark_arrays_handled_safely():
    # Empty list
    empty_vec = build_landmark_vector([])
    assert empty_vec.shape == (0,)

    # 2D coordinates (missing z dimension)
    points_2d = [[0.1 * i, 0.2 * i] for i in range(21)]
    vec_2d = build_landmark_vector(points_2d)
    assert vec_2d.shape[0] > 0

    # Fewer than 21 landmarks (e.g. 5 points)
    few_points = [[0.1, 0.2, 0.3] for _ in range(5)]
    vec_few = build_landmark_vector(few_points)
    assert vec_few.shape[0] > 0

    # Non-convertible / string elements in landmark array
    bad_points = [["invalid", "data", "test"] for _ in range(21)]
    vec_bad = build_landmark_vector(bad_points)
    assert vec_bad.shape == (0,)


def test_missing_landmarks_and_empty_windows(tmp_path):
    classifier = GestureClassifier(template_store_path=tmp_path / "empty_win.json")
    # Empty windows sequence
    token, conf = classifier.predict([])
    assert token == "unknown"
    assert conf == 0.0

    diag = classifier.predict_with_diagnostics([])
    assert diag["reason"] == "no_windows"
    assert diag["token"] == "unknown"


def test_non_finite_coordinates_produce_unknown_prediction(tmp_path):
    classifier = GestureClassifier(template_store_path=tmp_path / "non_finite.json")
    sample = make_landmark_sample(0.8)
    classifier.record_template("test_sign", [sample, sample])

    # Landmarks with NaN
    nan_sample = make_landmark_sample(0.8)
    nan_sample[5] = [float("nan"), 0.5, 0.0]
    token_nan, conf_nan = classifier.predict([nan_sample])
    assert token_nan == "unknown"
    assert conf_nan == 0.0

    diag_nan = classifier.predict_with_diagnostics([nan_sample])
    assert diag_nan["reason"] == "empty_or_degenerate_features"

    # Landmarks with Inf
    inf_sample = make_landmark_sample(0.8)
    inf_sample[3] = [float("inf"), 0.5, 0.0]
    token_inf, conf_inf = classifier.predict([inf_sample])
    assert token_inf == "unknown"
    assert conf_inf == 0.0

    # Normalization of non-finite landmarks returns safe zeros without crashing
    norm_nan = normalize_landmarks(nan_sample)
    assert norm_nan.shape == (21, 3)
    assert np.all(norm_nan == 0.0)


def test_empty_templates_prediction_and_recording_validation(tmp_path):
    classifier = GestureClassifier(template_store_path=tmp_path / "no_tmpl.json")
    # Explicitly clear templates to simulate empty state
    classifier.templates.clear()

    sample = make_landmark_sample(0.5)
    diag = classifier.predict_with_diagnostics([sample])
    assert diag["token"] == "unknown"
    assert diag["confidence"] == 0.0
    assert diag["reason"] == "no_templates"

    # Recording template with empty samples list raises ValueError
    with pytest.raises(ValueError, match="At least one valid sample is required"):
        classifier.record_template("open_hand", [])

    # Recording template with empty or whitespace label raises ValueError
    with pytest.raises(ValueError, match="Gesture label is required"):
        classifier.record_template("", [sample])

    with pytest.raises(ValueError, match="Gesture label is required"):
        classifier.record_template("   ", [sample])


def test_corrupted_template_storage_does_not_crash(tmp_path):
    # Corrupted JSON syntax
    bad_json_path = tmp_path / "syntax_error.json"
    bad_json_path.write_text("{invalid json content!!!", encoding="utf-8")
    c1 = GestureClassifier(template_store_path=bad_json_path)
    assert isinstance(c1.templates, dict)

    # Corrupted non-dict root
    list_json_path = tmp_path / "list_root.json"
    list_json_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    c2 = GestureClassifier(template_store_path=list_json_path)
    assert isinstance(c2.templates, dict)

    # Corrupted template vectors (strings instead of numbers)
    corrupted_vector_path = tmp_path / "corrupted_vec.json"
    corrupted_vector_path.write_text(
        json.dumps({
            "_meta": {"schema_version": 2},
            "templates": {
                "corrupted": {"vector": ["not", "numbers"], "feature_version": 2},
                "valid": {"vector": [0.1] * 82, "feature_version": 2},
            }
        }),
        encoding="utf-8",
    )
    c3 = GestureClassifier(template_store_path=corrupted_vector_path)
    assert "corrupted" not in c3.templates
    assert "valid" in c3.templates

    # Corrupted template vectors with non-finite values
    nan_vector_path = tmp_path / "nan_vec.json"
    nan_vector_path.write_text(
        json.dumps({
            "_meta": {"schema_version": 2},
            "templates": {
                "nan_gesture": {"vector": [None, 1.0, 2.0], "feature_version": 2},
            }
        }),
        encoding="utf-8",
    )
    c4 = GestureClassifier(template_store_path=nan_vector_path)
    assert "nan_gesture" not in c4.templates


def test_template_persistence_and_reload(tmp_path):
    store_file = tmp_path / "persistent_gestures.json"
    classifier1 = GestureClassifier(template_store_path=store_file)
    sample = make_landmark_sample(0.75)
    classifier1.record_template("wave", [sample, sample])

    assert store_file.exists()
    assert "wave" in classifier1.template_names()

    # Create new classifier instance pointing to the same file
    classifier2 = GestureClassifier(template_store_path=store_file)
    assert "wave" in classifier2.template_names()
    token, conf = classifier2.predict([sample])
    assert token == "wave"
    assert conf >= 0.60

    # Delete template and verify persistence
    deleted = classifier2.delete_template("wave")
    assert deleted is True
    assert "wave" not in classifier2.template_names()

    classifier3 = GestureClassifier(template_store_path=store_file)
    assert "wave" not in classifier3.template_names()


def test_unknown_gestures_and_low_confidence_predictions(tmp_path):
    classifier = GestureClassifier(
        template_store_path=tmp_path / "low_conf.json",
        low_confidence_threshold=0.85,
    )
    sample = make_landmark_sample(0.5)
    classifier.record_template("calibrated_pose", [sample, sample])

    # A completely different pose (e.g. inverted and scaled)
    novel_pose = [[-point[0] * 3.0, point[1] * 0.1, -point[2]] for point in sample]
    diag = classifier.predict_with_diagnostics([novel_pose])
    assert diag["token"] == "unknown"
    assert diag["confidence"] < 0.85
    assert diag["reason"] in ("low_confidence", "distance_ambiguity", "confidence_margin")


def test_append_caption_word_edge_cases_and_history_limits():
    # Empty or whitespace or None tokens do not alter history
    assert append_caption_word(["hello"], "") == ["hello"]
    assert append_caption_word(["hello"], "   ") == ["hello"]
    assert append_caption_word(["hello"], None) == ["hello"]

    # Starting from empty history
    assert append_caption_word([], "first") == ["first"]

    # Non-consecutive duplicates are permitted
    seq = []
    seq = append_caption_word(seq, "apple")
    seq = append_caption_word(seq, "banana")
    seq = append_caption_word(seq, "apple")
    assert seq == ["apple", "banana", "apple"]

    # Boundary tests for max_visible
    # Exactly max_visible
    base = [f"word_{i}" for i in range(5)]
    res = append_caption_word(base, "word_5", max_visible=6)
    assert len(res) == 6
    assert res[-1] == "word_5"

    # Exceeding max_visible by 1: oldest is trimmed
    res_trimmed = append_caption_word(res, "word_6", max_visible=6)
    assert len(res_trimmed) == 6
    assert res_trimmed[0] == "word_1"
    assert res_trimmed[-1] == "word_6"

    # Custom max_visible = 1
    single = append_caption_word(["first"], "second", max_visible=1)
    assert single == ["second"]
    single_dup = append_caption_word(single, "second", max_visible=1)
    assert single_dup == ["second"]
