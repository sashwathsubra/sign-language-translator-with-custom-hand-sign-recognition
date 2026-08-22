import numpy as np

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
