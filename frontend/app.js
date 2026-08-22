import { shouldAppendCaptionLabel } from './demo_logic.js';
import { GestureMatcher } from './matcher.js';

const matcher = new GestureMatcher();

const video = document.getElementById('video');
const canvas = document.getElementById('overlayCanvas');
const caption = document.getElementById('caption');
const confidence = document.getElementById('confidence');
const connectionState = document.getElementById('connectionState');
const statusMessage = document.getElementById('statusMessage');
const startDemo = document.getElementById('startDemo');
const startHero = document.getElementById('startHero');
const stopDemo = document.getElementById('stopDemo');
const resetDemo = document.getElementById('resetDemo');
const gestureLabel = document.getElementById('gestureLabel');
const recordButton = document.getElementById('recordGesture');
const starterButton = document.getElementById('starterVocabulary');
const trainedList = document.getElementById('trainedList');
const refreshTemplatesButton = document.getElementById('refreshTemplates');

let mediaStream = null;
let animationFrameId = null;
let isRunning = false;
let lastVideoTime = -1;
let lastCaption = '';
let captionHistory = [];
let lastEmittedLabel = null;
let pendingCaptionLabel = null;
let pendingCaptionFrames = 0;
let handLandmarker = null;
let isRecordingGesture = false;
let gestureSamples = [];
let trainingTimer = null;
let canvasContext = null;
let canvasWidth = 0;
let canvasHeight = 0;
let rVFCHandle = null;
let lastHandResult = null;

const MEDIA_PIPE_VERSION = '0.10.14';
const MEDIA_PIPE_CDN_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIA_PIPE_VERSION}`;
const backendUrl = (window.__BACKEND_URL__ || window.location.origin).replace(/\/$/, '');
const MAX_CAPTION_HISTORY = 20;
const CAPTION_STABILITY_FRAMES = 5;
const CAMERA_CONSTRAINTS_LOW_LATENCY = {
  video: {
    facingMode: 'user',
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30 },
  },
  audio: false,
};
const CAMERA_CONSTRAINTS_FALLBACK = {
  video: true,
  audio: false,
};
const ENABLE_MATCH_DEBUG_LOG = true;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], [18, 17]
];

let currentStatusState = '';
let currentStatusMessage = '';

function updateStatus(state, message) {
  if (currentStatusState !== state) {
    currentStatusState = state;
    if (connectionState) connectionState.textContent = state;
  }
  if (currentStatusMessage !== message) {
    currentStatusMessage = message;
    if (statusMessage) statusMessage.textContent = message;
  }
}

function setCaption(text) {
  const safeText = text || 'Waiting for a sign…';
  caption.textContent = safeText;
  lastCaption = safeText;
}

function updateCaptionHistory(token) {
  const label = String(token || '').trim();
  if (!label || !shouldAppendCaptionLabel(label)) {
    // Unknown/empty prediction — clear the pending label AND the last emitted label
    // so the floating canvas banner disappears immediately.
    pendingCaptionLabel = null;
    pendingCaptionFrames = 0;
    lastEmittedLabel = null;
    return;
  }

  if (pendingCaptionLabel === label) {
    pendingCaptionFrames += 1;
  } else {
    pendingCaptionLabel = label;
    pendingCaptionFrames = 1;
  }

  if (pendingCaptionFrames < CAPTION_STABILITY_FRAMES) {
    return;
  }

  if (lastEmittedLabel === label) {
    pendingCaptionLabel = null;
    pendingCaptionFrames = 0;
    return;
  }

  captionHistory = captionHistory.concat(label);
  captionHistory = captionHistory.slice(-MAX_CAPTION_HISTORY);
  lastEmittedLabel = label;
  pendingCaptionLabel = null;
  pendingCaptionFrames = 0;
  setCaption(captionHistory.join(' '));
}

function renderTemplateList(gestures) {
  trainedList.innerHTML = '';
  if (!gestures.length) {
    const item = document.createElement('li');
    item.textContent = 'No gestures recorded yet.';
    trainedList.appendChild(item);
    return;
  }

  gestures.forEach((gesture) => {
    const item = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = gesture.needs_rerecord
      ? `${gesture.label} (re-record recommended)`
      : gesture.label;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = 'Delete';
    remove.addEventListener('click', () => {
      matcher.deleteTemplate(gesture.label);
      fetchTemplates();
    });
    item.appendChild(label);
    item.appendChild(remove);
    trainedList.appendChild(item);
  });
}

async function fetchTemplates() {
  try {
    const localGestures = JSON.parse(localStorage.getItem('slt_gestures') || '{}');
    const gestures = Object.keys(localGestures).map(label => ({
      label,
      count: 1,
      needs_rerecord: false
    }));
    renderTemplateList(gestures);
    maybeShowStarterGuide(gestures);
  } catch (error) {
    console.error(error);
  }
}



function drawLandmarks(handSet, handednesses) {
  if (!canvasContext) {
    canvasContext = canvas.getContext('2d', { desynchronized: true, alpha: true });
  }
  const ctx = canvasContext;
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  if (canvasWidth !== width || canvasHeight !== height) {
    canvas.width = width;
    canvas.height = height;
    canvasWidth = width;
    canvasHeight = height;
  }
  ctx.clearRect(0, 0, width, height);

  const hands = Array.isArray(handSet) && handSet.length && Array.isArray(handSet[0]) ? handSet : [handSet].filter(Array.isArray);
  if (!hands.length || !hands[0].length) return;

  // Build touching point map (avoid string keys — use typed array index)
  const touchingPoints = new Uint8Array(42);
  if (hands.length >= 2) {
    for (let i = 0; i < hands[0].length; i++) {
      for (let j = 0; j < hands[1].length; j++) {
        const p1 = hands[0][i];
        const p2 = hands[1][j];
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dz = p1.z - p2.z;
        if (dx * dx + dy * dy + dz * dz < 0.0025) {
          touchingPoints[i] = 1;
          touchingPoints[21 + j] = 1;
        }
      }
    }
  }

  hands.forEach((landmarks, index) => {
    if (!landmarks || !landmarks.length) return;

    let isRight = false;
    let isPrimary = false;
    if (handednesses && handednesses[index] && handednesses[index][0]) {
      const cat = handednesses[index][0].categoryName || handednesses[index][0].displayName;
      isRight = (cat === 'Right');
    }
    
    // Determine if this is the primary hand (Right hand, or the only hand)
    if (hands.length === 1) {
      isPrimary = true;
    } else {
      isPrimary = isRight; // Default to right hand being primary when both are visible
    }

    ctx.strokeStyle = (hands.length === 1 || isRight) ? '#38bdf8' : '#a78bfa';
    ctx.lineWidth = 2;

    // Draw skeleton lines in one batch
    ctx.beginPath();
    for (const [start, end] of HAND_CONNECTIONS) {
      const p1 = landmarks[start];
      const p2 = landmarks[end];
      if (!p1 || !p2) continue;
      const x1 = (p1.x || 0) * width;
      const y1 = (p1.y || 0) * height;
      const x2 = (p2.x || 0) * width;
      const y2 = (p2.y || 0) * height;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // Draw normal dots in one batch, then touching dots in another
    const touchOffset = index * 21;
    ctx.fillStyle = '#f8fafc';
    ctx.beginPath();
    for (let pIndex = 0; pIndex < landmarks.length; pIndex++) {
      if (touchingPoints[touchOffset + pIndex]) continue;
      const p = landmarks[pIndex];
      ctx.moveTo((p.x || 0) * width + 3, (p.y || 0) * height);
      ctx.arc((p.x || 0) * width, (p.y || 0) * height, 3, 0, Math.PI * 2);
    }
    ctx.fill();

    ctx.fillStyle = 'red';
    ctx.beginPath();
    for (let pIndex = 0; pIndex < landmarks.length; pIndex++) {
      if (!touchingPoints[touchOffset + pIndex]) continue;
      const p = landmarks[pIndex];
      ctx.moveTo((p.x || 0) * width + 3, (p.y || 0) * height);
      ctx.arc((p.x || 0) * width, (p.y || 0) * height, 4, 0, Math.PI * 2);
    }
    ctx.fill();

    // Use lastEmittedLabel (the stable committed prediction) to drive the banner.
    // It is explicitly set to null when the matcher returns 'unknown', so the
    // banner clears immediately on unknown gestures.
    const currentLabel = lastEmittedLabel;
    
    if (isPrimary) {
      // Calculate wrist angle visually (roll)
      const wrist = landmarks[0];
      const indexMcp = landmarks[5];
      const pinkyMcp = landmarks[17];
      let rollDeg = 0;
      if (wrist && indexMcp && pinkyMcp) {
        rollDeg = Math.round(Math.atan2(indexMcp.y - pinkyMcp.y, indexMcp.x - pinkyMcp.x) * (180 / Math.PI));
        ctx.fillStyle = '#fcd34d'; // yellow
        ctx.font = '10px "DM Sans", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${rollDeg}°`, (wrist.x * width) + 10, (wrist.y * height));
      }

      if (currentLabel && currentLabel !== 'unknown') {
        let minX = 1;
        let maxX = 0;
        let minY = 1;
        for (const p of landmarks) {
          if (p.x < minX) minX = p.x;
          if (p.x > maxX) maxX = p.x;
          if (p.y < minY) minY = p.y;
        }
        const centerX = (minX + maxX) / 2;

        const text = currentLabel.toUpperCase();
        ctx.font = 'bold 12px "DM Sans", sans-serif'; // even smaller font
        const textMetrics = ctx.measureText(text);
        const paddingX = 8;
        const paddingY = 4;
        const bannerW = textMetrics.width + (paddingX * 2);
        const bannerH = 20;

        const bx = (centerX * width) - (bannerW / 2);
        const by = (minY * height) - bannerH - 60; // Float well above the hand (60px clearance)

        ctx.fillStyle = 'rgba(18, 19, 21, 0.85)'; 
        ctx.beginPath();
        ctx.roundRect(bx, by, bannerW, bannerH, 4);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, bx + (bannerW / 2), by + (bannerH / 2));
      }
    }
  });
}

function pickClassifierHand(result) {
  const hands = result?.landmarks || [];
  if (!hands.length) {
    return null;
  }

  const handedness = result?.handednesses || result?.handedness || [];
  let bestIndex = 0;
  let bestScore = -1;
  for (let i = 0; i < hands.length; i += 1) {
    const score = Number(handedness?.[i]?.[0]?.score ?? 0);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }
  return hands[bestIndex] || null;
}

function buildFrameContext(hand) {
  if (!Array.isArray(hand) || hand.length < 21) {
    return {};
  }

  const wrist = hand[0];
  const middleMcp = hand[9];
  const indexMcp = hand[5];
  const pinkyMcp = hand[17];

  const distance = (a, b) => Math.hypot((a.x - b.x), (a.y - b.y), (a.z - b.z));
  const scale = distance(wrist, middleMcp);
  const roll = Math.atan2(indexMcp.y - pinkyMcp.y, indexMcp.x - pinkyMcp.x) * (180 / Math.PI);
  const tilt = Math.atan2(Math.abs(middleMcp.z - wrist.z), Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y)) * (180 / Math.PI);

  let distanceBand = 'mid';
  if (scale < 0.08) distanceBand = 'far';
  if (scale > 0.16) distanceBand = 'near';

  return {
    scale: Number(scale.toFixed(4)),
    roll_deg: Number(roll.toFixed(2)),
    tilt_deg: Number(tilt.toFixed(2)),
    distance_band: distanceBand,
  };
}

async function createHandLandmarker() {
  if (handLandmarker) return handLandmarker;

  try {
    const bundleUrl = `${MEDIA_PIPE_CDN_ROOT}/vision_bundle.mjs`;
    const { FilesetResolver, HandLandmarker } = await import(bundleUrl);
    const vision = await FilesetResolver.forVisionTasks(`${MEDIA_PIPE_CDN_ROOT}/wasm`);
    const modelAssetPath = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    return handLandmarker;
  } catch (error) {
    console.error('MediaPipe model initialization failed:', error.message || error, {
      originalError: error,
      cdnRoot: MEDIA_PIPE_CDN_ROOT,
      wasmPath: `${MEDIA_PIPE_CDN_ROOT}/wasm`,
      modelPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    });
    throw error;
  }
}

async function getCameraStreamWithFallback() {
  try {
    return await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS_LOW_LATENCY);
  } catch (error) {
    console.warn('Low-latency camera constraints failed, retrying with default constraints.', {
      name: error?.name,
      message: error?.message,
    });
    return navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS_FALLBACK);
  }
}

function debugSendRate(nowMs) {
  const elapsed = nowMs - sendWindowStart;
  if (elapsed >= 1000) {
    const sendsPerSecond = sendCounter / (elapsed / 1000);
    console.log('[demo send-rate]', sendsPerSecond.toFixed(2), 'msgs/sec');
    sendCounter = 0;
    sendWindowStart = nowMs;
  }
}

function detectHandFrame() {
  if (!video.videoWidth || !video.videoHeight || !handLandmarker) return;

  const now = performance.now();
  let result;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      result = handLandmarker.detectForVideo(video, now);
      lastHandResult = result;
    } catch (e) {
      console.error('MediaPipe detection error:', e);
      return;
    }
  } else {
    result = lastHandResult;
  }

  if (!result) return;

  const hands = result?.landmarks || [];
  const handednesses = result?.handednesses || result?.handedness || [];

  if (!hands.length) {
    drawLandmarks([], []);
    confidence.textContent = 'Confidence: —';
    updateStatus('Idle', 'No hand detected. Show your hand in frame to begin.');
    return;
  }

  drawLandmarks(hands, handednesses);
  updateStatus('Connected', 'Tracking hand.');

  let normalizedClassifierHand = [];

  if (hands.length >= 2) {
    let primaryIdx = 0;
    if (handednesses[1] && handednesses[1][0] && handednesses[1][0].categoryName === 'Right') {
       primaryIdx = 1;
    } else if (handednesses[0] && handednesses[0][0] && handednesses[0][0].categoryName === 'Right') {
       primaryIdx = 0;
    }
    const classifierHand = hands[primaryIdx];
    normalizedClassifierHand = classifierHand.map((point) => [point.x, point.y, point.z]);
  } else {
    const classifierHand = pickClassifierHand(result);
    if (!classifierHand) return;
    normalizedClassifierHand = classifierHand.map((point) => [point.x, point.y, point.z]);
  }

  // ── Client-side classification (no WebSocket round-trip) ──────────────────
  if (normalizedClassifierHand.length >= 21) {
    const prediction = matcher.predict(normalizedClassifierHand);
    updateCaptionHistory(prediction.token);
    const confPct = prediction.token !== 'unknown'
      ? `Confidence: ${Math.round(prediction.confidence * 100)}%`
      : 'Confidence: —';
    confidence.textContent = confPct;
    if (ENABLE_MATCH_DEBUG_LOG && prediction.top_candidates?.length) {
      console.debug('[match-debug]', prediction);
    }
  }

  if (isRecordingGesture) {
    gestureSamples.push(normalizedClassifierHand);
  }
}

async function startCamera() {
  if (isRunning) return;
  isRunning = true;
  updateStatus('Connecting…', 'Requesting camera access.');

  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API not supported');
    }
    mediaStream = await getCameraStreamWithFallback();
  } catch (error) {
    isRunning = false;
    console.error('Camera permission error:', error);

    if (error && error.name === 'NotAllowedError') {
      updateStatus('Camera blocked', 'Camera permission was denied. Please allow webcam access in the browser and try again.');
    } else if (error && error.name === 'NotFoundError') {
      updateStatus('Camera unavailable', 'No camera was found on this device. Please connect a webcam and try again.');
    } else if (error && error.name === 'NotReadableError') {
      updateStatus('Camera busy', 'The camera is already in use by another app or browser tab. Close it and retry.');
    } else {
      updateStatus('Camera unavailable', 'Camera access could not be granted. Please check browser permissions and try again.');
    }
    return;
  }

  try {
    video.srcObject = mediaStream;
    await video.play();
    updateStatus('Camera ready', 'Camera is active. Initializing hand tracking model.');
  } catch (error) {
    isRunning = false;
    console.error('Video playback failed after camera access was granted:', error);
    updateStatus('Camera unavailable', 'Camera started but video playback failed. Close other camera apps/tabs and retry.');
    setCaption('Camera playback unavailable');
    return;
  }

  try {
    await createHandLandmarker();
  } catch (error) {
    updateStatus('Model load failed', 'Camera is on, but hand-tracking model failed to load. Check console for CDN/model URL errors.');
    setCaption('Hand-tracking unavailable (camera still active)');
  }

  try {
    const tick = () => {
      if (!isRunning) return;
      detectHandFrame();
      if (video.requestVideoFrameCallback) {
        rVFCHandle = video.requestVideoFrameCallback(tick);
      } else {
        animationFrameId = window.requestAnimationFrame(tick);
      }
    };
    if (video.requestVideoFrameCallback) {
      rVFCHandle = video.requestVideoFrameCallback(tick);
    } else {
      animationFrameId = window.requestAnimationFrame(tick);
    }
    updateStatus('Connected', 'Gesture recognition is fully client-side.');
    setCaption('Waiting for a hand…');
  } catch (error) {
    isRunning = false;
    console.error('Streaming setup failed:', error);
    updateStatus('Streaming error', 'Camera and model are ready, but live streaming setup failed. Retry start.');
    setCaption('Streaming unavailable');
  }
}

function stopCamera() {
  isRunning = false;
  if (animationFrameId) {
    window.cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (rVFCHandle && video.cancelVideoFrameCallback) {
    video.cancelVideoFrameCallback(rVFCHandle);
    rVFCHandle = null;
  }
  rVFCHandle = null;
  lastLandmarkSendAt = 0;
  sendCounter = 0;
  sendWindowStart = performance.now();
  lastVideoTime = -1;
  if (trainingTimer) {
    window.clearTimeout(trainingTimer);
    trainingTimer = null;
  }
  handLandmarker = null;
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (video.srcObject) {
    video.srcObject = null;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setCaption('Waiting for a sign…');
  confidence.textContent = 'Confidence: —';
  updateStatus('Idle', 'Ready to start again.');
}

async function recordGesture(labelOverride = '') {
  const label = (labelOverride || (gestureLabel.value || '')).trim();
  if (!label) {
    updateStatus('Needs label', 'Type a name before recording a gesture.');
    return false;
  }
  if (!isRunning) {
    updateStatus('Start camera', 'Turn on the camera before recording a sign.');
    return false;
  }
  if (isRecordingGesture) {
    return false;
  }

  isRecordingGesture = true;
  gestureSamples = [];
  updateStatus('Recording…', `Hold ${label} for 2–3 seconds.`);
  trainingTimer = window.setTimeout(() => {
    isRecordingGesture = false;
    if (gestureSamples.length === 0) {
      updateStatus('Error', 'No frames captured. Is the hand visible?');
      return;
    }
    const samples = gestureSamples.slice(-20);
    try {
      matcher.recordTemplate(label, samples); // saves to localStorage internally
      updateStatus('Saved', `${label} was added to the trained vocabulary.`);
      fetchTemplates();
    } catch (error) {
      updateStatus('Error', error.message || 'Unable to save the gesture.');
      console.error(error);
    }
  }, 2500);
  return true;
}

async function runStarterVocabularyFlow() {
  const starterLabels = ['hello', 'yes', 'no', 'thank_you', 'stop', 'ok', 'wave', 'fist', 'open_hand', 'point', 'thumbs_up', 'gun'];
  if (!isRunning) {
    updateStatus('Start camera', 'Turn on the camera before recording the starter vocabulary.');
    return;
  }

  for (let index = 0; index < starterLabels.length; index += 1) {
    const label = starterLabels[index];
    gestureLabel.value = label;
    updateStatus('Starter setup', `Record starter gesture ${index + 1}/${starterLabels.length}: ${label}`);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    await recordGesture(label);
    await new Promise((resolve) => window.setTimeout(resolve, 2700));
    if (isRecordingGesture) {
      await new Promise((resolve) => window.setTimeout(resolve, 600));
    }
  }

  updateStatus('Starter set ready', 'Starter vocabulary recorded. You can keep adding your own signs any time.');
}

function maybeShowStarterGuide(gestures) {
  if (gestures && gestures.length >= 3) {
    return;
  }

  const starterLabels = ['hello', 'yes', 'no', 'thank_you', 'stop', 'ok', 'wave', 'fist', 'open_hand', 'point', 'thumbs_up'];
  updateStatus('Starter setup', `No trained gestures yet. Record a few starter signs: ${starterLabels.slice(0, 6).join(', ')}.`);
}

function bindEvents() {
  if (startDemo) startDemo.addEventListener('click', startCamera);
  if (startHero) startHero.addEventListener('click', startCamera);
  if (stopDemo) stopDemo.addEventListener('click', stopCamera);
  if (resetDemo) resetDemo.addEventListener('click', stopCamera);
  if (recordButton) recordButton.addEventListener('click', () => recordGesture());
  if (starterButton) starterButton.addEventListener('click', runStarterVocabularyFlow);
  if (refreshTemplatesButton) refreshTemplatesButton.addEventListener('click', fetchTemplates);
  window.addEventListener('beforeunload', stopCamera);
}

window.addEventListener('load', () => {
  fetchTemplates();
});

bindEvents();
