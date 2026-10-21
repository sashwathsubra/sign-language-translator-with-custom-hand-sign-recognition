import json
import time

from fastapi.testclient import TestClient

from sign_language_translator.demo.pipeline import DemoPipeline
from server import DemoSession, MAX_CONCURRENT_SESSIONS, active_sessions, app


def make_valid_payload():
    return {
        "landmarks": [
            [0.1, 0.2, 0.0],
            [0.2, 0.3, 0.0],
            [0.3, 0.4, 0.0],
        ]
    }


def test_malformed_payload_is_rejected_without_crashing():
    session = DemoSession(DemoPipeline())

    result = session.handle_payload({"landmarks": [[0.1, 0.2, None]]})

    assert result["ok"] is False
    assert result["reason"] == "invalid_payload"


def test_frame_buffer_overflow_is_dropped_gracefully():
    session = DemoSession(DemoPipeline(), max_buffer_size=2)

    first = session.handle_payload(make_valid_payload())
    second = session.handle_payload(make_valid_payload())
    third = session.handle_payload(make_valid_payload())

    assert first["ok"] is True
    assert second["ok"] is True
    assert third["ok"] is False
    assert third["reason"] == "buffer_full"


def test_idle_session_is_marked_for_cleanup():
    session = DemoSession(DemoPipeline(), idle_timeout_seconds=0.5)
    session.last_activity_at = time.monotonic() - 1.0

    assert session.should_close() is True


def test_no_hand_payload_clears_buffer_and_skips_classification():
    session = DemoSession(DemoPipeline(), max_buffer_size=4)
    session.handle_payload({"hand_detected": True, "landmarks": make_valid_payload()["landmarks"]})

    result = session.handle_payload({"hand_detected": False})

    assert result["ok"] is False
    assert result["reason"] == "no_hand_detected"
    assert len(session.buffer) == 0


def test_buffer_overflow_recovery_after_rate_normalizes():
    session = DemoSession(DemoPipeline(), max_buffer_size=2)
    overflow_seen = False

    for _ in range(10):
        result = session.handle_payload(make_valid_payload())
        if result["ok"] is False and result["reason"] == "buffer_full":
            overflow_seen = True
            break

    assert overflow_seen is True
    assert len(session.buffer) == 0

    recovery = session.handle_payload(make_valid_payload())
    assert recovery["ok"] is True
    assert len(session.buffer) == 1


def test_recording_endpoint_stores_and_lists_templates():
    client = TestClient(app)
    samples = []
    for idx in range(3):
        samples.append([
            [float(idx) / 20.0, float(idx) / 30.0, 0.0] for _ in range(21)
        ])

    response = client.post(
        "/api/gestures/record",
        json={"label": "hello", "samples": samples},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "hello"
    assert body["stored"] is True

    list_response = client.get("/api/gestures")
    payload = list_response.json()
    assert any(item["label"] == "hello" for item in payload["gestures"])


def test_prediction_requires_stable_consecutive_frames():
    pipeline = DemoPipeline()
    sample = [[float(i) / 30.0, float(i % 4) / 40.0, 0.0] for i in range(21)]
    pipeline.classifier.record_template("steady", [sample, sample])
    session = DemoSession(pipeline)

    first = session.handle_payload({"hand_detected": True, "landmarks": sample})
    second = session.handle_payload({"hand_detected": True, "landmarks": sample})
    third = session.handle_payload({"hand_detected": True, "landmarks": sample})

    assert first["ok"] is True and first["token"] == "unknown"
    assert second["ok"] is True and second["token"] == "unknown"
    assert third["ok"] is True and third["token"] == "steady"


def test_non_dict_payload_rejected():
    session = DemoSession(DemoPipeline())
    assert session.handle_payload(None) == {"ok": False, "reason": "invalid_payload"}
    assert session.handle_payload("invalid") == {"ok": False, "reason": "invalid_payload"}
    assert session.handle_payload([1, 2, 3]) == {"ok": False, "reason": "invalid_payload"}


def test_payload_with_missing_landmarks_when_hand_detected():
    session = DemoSession(DemoPipeline())
    res1 = session.handle_payload({"hand_detected": True, "landmarks": []})
    assert res1 == {"ok": False, "reason": "invalid_payload"}

    res2 = session.handle_payload({"hand_detected": True})
    assert res2 == {"ok": False, "reason": "invalid_payload"}


def test_payload_with_non_finite_coordinates_rejected():
    session = DemoSession(DemoPipeline())
    res_nan = session.handle_payload({"landmarks": [[float("nan"), 0.2, 0.0]]})
    assert res_nan == {"ok": False, "reason": "invalid_payload"}

    res_inf = session.handle_payload({"landmarks": [[0.1, float("inf"), 0.0]]})
    assert res_inf == {"ok": False, "reason": "invalid_payload"}

    res_neginf = session.handle_payload({"landmarks": [[0.1, float("-inf"), 0.0]]})
    assert res_neginf == {"ok": False, "reason": "invalid_payload"}


def test_nested_multihand_payload_validation():
    session = DemoSession(DemoPipeline())
    hand = [[0.1 * i, 0.05 * (i % 3), 0.0] for i in range(21)]
    valid_multihand = {"landmarks": [hand, hand]}
    res_valid = session.handle_payload(valid_multihand)
    assert res_valid["ok"] is True

    bad_multihand = {"landmarks": [hand, [[0.1, float("nan"), 0.0] for _ in range(21)]]}
    res_bad = session.handle_payload(bad_multihand)
    assert res_bad == {"ok": False, "reason": "invalid_payload"}


def test_websocket_greeting_and_malformed_messages():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        greeting = ws.receive_json()
        assert greeting["type"] == "status"
        assert "Connected" in greeting["message"]

        # Send malformed non-JSON string
        ws.send_text("this is not json {")
        reply = ws.receive_json()
        assert reply["type"] == "status"
        assert reply["message"] == "The server received an invalid message."

        # Send malformed landmark structure
        ws.send_json({"landmarks": "invalid_string_not_list"})
        reply2 = ws.receive_json()
        assert reply2["type"] == "status"
        assert reply2["message"] == "Received malformed landmark data."

        # Send no hand detected
        ws.send_json({"hand_detected": False})
        reply3 = ws.receive_json()
        assert reply3["type"] == "status"
        assert reply3["message"] == "No hand detected. Waiting for a sign."


def test_websocket_unexpected_processing_error_handled_gracefully(monkeypatch):
    client = TestClient(app)

    def failing_handle_payload(self, payload):
        raise RuntimeError("Simulated internal pipeline error")

    monkeypatch.setattr(DemoSession, "handle_payload", failing_handle_payload)

    with client.websocket_connect("/ws") as ws:
        greeting = ws.receive_json()
        assert greeting["type"] == "status"

        ws.send_json({"hand_detected": True, "landmarks": [[0.1, 0.2, 0.0]]})
        reply = ws.receive_json()
        assert reply["type"] == "error"
        assert reply["message"] == "An error occurred while processing the frame."


def test_websocket_busy_when_max_concurrent_sessions():
    client = TestClient(app)
    dummy_sessions = [DemoSession(DemoPipeline()) for _ in range(MAX_CONCURRENT_SESSIONS)]
    for s in dummy_sessions:
        active_sessions.add(s)

    try:
        with client.websocket_connect("/ws") as ws:
            reply = ws.receive_json()
            assert reply["type"] == "error"
            assert "Server is busy" in reply["message"]
    finally:
        for s in dummy_sessions:
            active_sessions.discard(s)
