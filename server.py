import json
import logging
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from sign_language_translator.demo.pipeline import DemoPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("slt_demo")

app = FastAPI(title="SLT Demo Server")
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gesture_store_path = Path(os.getenv("SLT_DEMO_GESTURE_STORE", ".demo_gestures.json"))
pipeline = DemoPipeline(gesture_store_path)

MAX_BUFFER_SIZE = int(os.getenv("MAX_BUFFER_SIZE", "16"))
IDLE_TIMEOUT_SECONDS = float(os.getenv("IDLE_TIMEOUT_SECONDS", "60"))
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "32"))
MAX_TEMPLATE_COUNT = 50
MIN_STABLE_PREDICTION_FRAMES = int(os.getenv("MIN_STABLE_PREDICTION_FRAMES", "3"))
MIN_ACCEPTED_CONFIDENCE = float(os.getenv("MIN_ACCEPTED_CONFIDENCE", "0.68"))
NEAR_NEIGHBOR_EXTRA_STABLE_FRAMES = int(os.getenv("NEAR_NEIGHBOR_EXTRA_STABLE_FRAMES", "2"))
ENABLE_MATCH_DEBUG = os.getenv("ENABLE_MATCH_DEBUG", "1") == "1"

active_sessions: set["DemoSession"] = set()


class DemoSession:
    def __init__(
        self,
        pipeline: DemoPipeline,
        *,
        max_buffer_size: int = MAX_BUFFER_SIZE,
        idle_timeout_seconds: float = IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self.pipeline = pipeline
        self.session_id = str(uuid.uuid4())[:8]
        self.max_buffer_size = max_buffer_size
        self.idle_timeout_seconds = idle_timeout_seconds
        self.last_activity_at = time.monotonic()
        self.buffer: Deque[dict[str, Any]] = deque(maxlen=max_buffer_size)
        self.websocket: Optional[WebSocket] = None
        self.hand_detected = False
        self.last_token = "unknown"
        self.stable_frames = 0

    def clear_hand_state(self) -> None:
        self.hand_detected = False
        self.buffer.clear()
        self.last_token = "unknown"
        self.stable_frames = 0

    def _stabilize_prediction(self, token: str, confidence: float, diagnostics: Optional[dict[str, Any]] = None) -> tuple[str, float]:
        if token != self.last_token:
            self.last_token = token
            self.stable_frames = 1
        else:
            self.stable_frames += 1

        required_frames = MIN_STABLE_PREDICTION_FRAMES
        if diagnostics and diagnostics.get("confusable_pair"):
            required_frames += NEAR_NEIGHBOR_EXTRA_STABLE_FRAMES

        if token == "unknown" or confidence < MIN_ACCEPTED_CONFIDENCE:
            return "unknown", confidence
        if self.stable_frames < required_frames:
            return "unknown", confidence
        return token, confidence

    def should_close(self) -> bool:
        return (time.monotonic() - self.last_activity_at) > self.idle_timeout_seconds

    def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.last_activity_at = time.monotonic()
        if not isinstance(payload, dict):
            return {"ok": False, "reason": "invalid_payload"}

        if payload.get("hand_detected") is False:
            self.clear_hand_state()
            return {"ok": False, "reason": "no_hand_detected"}

        if payload.get("hand_detected") is True:
            self.hand_detected = True

        landmarks = payload.get("landmarks")
        if not isinstance(landmarks, list) or not landmarks:
            if self.hand_detected:
                return {"ok": False, "reason": "invalid_payload"}
            return {"ok": False, "reason": "no_hand_detected"}

        if len(self.buffer) >= self.max_buffer_size:
            self.clear_hand_state()
            return {"ok": False, "reason": "buffer_full"}

        normalized_landmarks: Any = []
        hand_sets: list[list[list[float]]] = []
        if isinstance(landmarks[0], list) and landmarks[0] and isinstance(landmarks[0][0], (list, tuple)):
            for hand in landmarks:
                if not isinstance(hand, list):
                    return {"ok": False, "reason": "invalid_payload"}
                hand_points: list[list[float]] = []
                for item in hand:
                    if not isinstance(item, list) or len(item) < 3:
                        return {"ok": False, "reason": "invalid_payload"}
                    try:
                        values = [float(value) for value in item]
                    except (TypeError, ValueError):
                        return {"ok": False, "reason": "invalid_payload"}
                    if any(not (value == value and value not in {float("inf"), float("-inf")}) for value in values):
                        return {"ok": False, "reason": "invalid_payload"}
                    hand_points.append(values)
                hand_sets.append(hand_points)
            normalized_landmarks = hand_sets
        else:
            normalized_landmarks = []
            for item in landmarks:
                if not isinstance(item, list) or len(item) < 3:
                    return {"ok": False, "reason": "invalid_payload"}
                try:
                    values = [float(value) for value in item]
                except (TypeError, ValueError):
                    return {"ok": False, "reason": "invalid_payload"}
                if any(not (value == value and value not in {float("inf"), float("-inf")}) for value in values):
                    return {"ok": False, "reason": "invalid_payload"}
                normalized_landmarks.append(values)

        self.buffer.append({"landmarks": normalized_landmarks})
        details = self.pipeline.classifier.predict_with_diagnostics([normalized_landmarks])
        token = str(details.get("token", "unknown"))
        confidence = float(details.get("confidence", 0.0))
        stable_token, stable_confidence = self._stabilize_prediction(token, confidence, details)

        if ENABLE_MATCH_DEBUG:
            logger.info(
                "match_debug session=%s token=%s stable_token=%s conf=%.3f top2=%s margin=%.3f reason=%s frame=%s",
                self.session_id,
                token,
                stable_token,
                confidence,
                details.get("top_candidates", []),
                float(details.get("confidence_margin", 0.0)),
                details.get("reason", "unknown"),
                payload.get("frame_context", {}),
            )

        sentence = self.pipeline.process_gloss(stable_token)
        return {
            "ok": True,
            "token": stable_token,
            "confidence": round(float(stable_confidence), 2),
            "sentence": sentence,
            "top_candidates": details.get("top_candidates", []),
            "match_reason": details.get("reason", "unknown"),
            "confidence_margin": round(float(details.get("confidence_margin", 0.0)), 5),
            "confusable_pair": bool(details.get("confusable_pair", False)),
        }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "demo-server", "sessions": len(active_sessions)}


@app.get("/api/gestures")
def list_gestures() -> dict[str, Any]:
    migration_status = pipeline.classifier.template_migration_status()
    gestures = [
        {
            "label": label,
            "count": 1,
            "needs_rerecord": bool(migration_status.get(label, False)),
        }
        for label in pipeline.classifier.template_names()
    ]
    return {"gestures": gestures, "max_templates": MAX_TEMPLATE_COUNT, "total": len(gestures)}


@app.post("/api/gestures/record")
def record_gesture(payload: dict[str, Any]) -> dict[str, Any]:
    label = str(payload.get("label", "")).strip()
    samples = payload.get("samples")
    if not label:
        raise HTTPException(status_code=400, detail="Gesture label is required.")
    if not isinstance(samples, list) or not samples:
        raise HTTPException(status_code=400, detail="At least one sample is required.")
    try:
        pipeline.classifier.record_template(label, samples)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "label": label,
        "stored": True,
        "templates": len(pipeline.classifier.templates),
        "max_templates": MAX_TEMPLATE_COUNT,
    }


@app.delete("/api/gestures/{label}")
def delete_gesture(label: str) -> dict[str, Any]:
    deleted = pipeline.classifier.delete_template(label)
    if not deleted:
        raise HTTPException(status_code=404, detail="Gesture template not found.")
    return {"deleted": True, "label": label}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    session = DemoSession(pipeline)
    if len(active_sessions) >= MAX_CONCURRENT_SESSIONS:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Server is busy. Please try again shortly."})
        await websocket.close()
        return

    active_sessions.add(session)
    await websocket.accept()
    session.websocket = websocket
    logger.info("websocket_open session=%s", session.session_id)
    await websocket.send_json(
        {
            "type": "status",
            "message": "Connected. Send real hand-landmark frames to receive captions.",
        }
    )

    try:
        while True:
            if session.should_close():
                logger.info("session_idle_close session=%s", session.session_id)
                break
            payload_text = await websocket.receive_text()
            session.last_activity_at = time.monotonic()
            try:
                data = json.loads(payload_text)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "status", "message": "The server received an invalid message."})
                continue

            result = session.handle_payload(data)
            if not result["ok"]:
                if result["reason"] == "buffer_full":
                    await websocket.send_json({"type": "status", "message": "The session is receiving data faster than it can be processed."})
                elif result["reason"] == "no_hand_detected":
                    await websocket.send_json({"type": "status", "message": "No hand detected. Waiting for a sign."})
                else:
                    await websocket.send_json({"type": "status", "message": "Received malformed landmark data."})
                continue

            logger.info(
                "prediction session=%s token=%s confidence=%.2f",
                session.session_id,
                result.get("token"),
                result.get("confidence", 0.0),
            )
            await websocket.send_json(
                {
                    "type": "prediction",
                    "token": result.get("token"),
                    "confidence": result.get("confidence"),
                    "sentence": result.get("sentence"),
                    "top_candidates": result.get("top_candidates", []),
                    "match_reason": result.get("match_reason"),
                    "confidence_margin": result.get("confidence_margin"),
                    "confusable_pair": result.get("confusable_pair", False),
                }
            )
    except Exception as exc:  # pragma: no cover - defensive isolation
        logger.exception("websocket_error session=%s error=%s", session.session_id, exc)
        try:
            await websocket.send_json({"type": "error", "message": "The connection closed unexpectedly."})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        active_sessions.discard(session)
        logger.info("websocket_close session=%s", session.session_id)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
