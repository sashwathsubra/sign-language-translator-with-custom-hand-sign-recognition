export function shouldAppendCaptionLabel(label) {
  const normalized = String(label || '').trim().toLowerCase();
  if (!normalized) return false;
  return !['unknown', 'no hand detected', 'waiting for a hand', 'waiting for a sign'].includes(normalized);
}

export function shouldSendLandmarks({ socketReady, now, lastLandmarkSendAt, intervalMs }) {
  if (!socketReady) return false;
  if (!Number.isFinite(lastLandmarkSendAt)) return true;
  return now - lastLandmarkSendAt >= intervalMs;
}
