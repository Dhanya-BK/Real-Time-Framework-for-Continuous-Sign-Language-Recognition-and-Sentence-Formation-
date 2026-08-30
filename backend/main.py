import os
import io
import json
import base64
import asyncio
import threading

import numpy as np
import cv2
import mediapipe as mp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

from dotenv import load_dotenv
from groq import Groq

from gtts import gTTS

from labels import GESTURES
from translator import TranslatorModule

# ============================================================
# SETUP — identical constants/logic to prediction.py
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "\nGROQ_API_KEY was not found.\n"
        "Create a .env file in the backend folder.\n"
        "Add:\nGROQ_API_KEY=your_actual_groq_api_key\n"
    )

GROQ_MODEL = "openai/gpt-oss-120b"
groq_client = Groq(api_key=GROQ_API_KEY)

translator = TranslatorModule()

MODEL_PATH = "gesture_model.h5"
SEQUENCE_LENGTH = 20
FEATURES = 225
THRESHOLD = 0.80
HISTORY_LENGTH = 5

MIN_BRIGHTNESS = 50
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

LANGUAGE_CODES = {
    "English": "en", "Kannada": "kn", "Hindi": "hi",
    "Tamil": "ta", "Telugu": "te", "Malayalam": "ml",
}

# ============================================================
# LOAD MODEL (via weights — sidesteps a Keras/TF version mismatch)
#
# load_model(MODEL_PATH) was tried and fails in this environment with:
#   ValueError: Unknown dtype policy: 'DTypePolicy'
# This happens because gesture_model.h5 was saved by a newer Keras
# version (Keras 3.x-style dtype policy serialization) than what's
# installed here, and load_model() needs that saved config to fully
# reconstruct the model. load_weights() only needs the raw weight
# tensors, not the serialized config/policy metadata, so it avoids
# this incompatibility entirely — which is why this manual rebuild
# approach is used instead of load_model().
#
# This DOES mean the architecture below must match the originally
# trained architecture exactly (same layers, same order, same units,
# same dropout placement) — load_weights() assigns weights layer-by-
# layer in order, not by name, so a mismatched reconstruction can
# silently load weights into the wrong layers with no error at all.
# If recognition accuracy seems off in ways that don't make sense,
# this architecture definition is the first thing worth re-verifying
# against however the model was originally trained/saved.
# ============================================================

print()
print("Loading gesture recognition model...")

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQUENCE_LENGTH, FEATURES)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(GESTURES), activation="softmax"))
model.load_weights(MODEL_PATH)

# Warm up so the first real prediction isn't slow
model.predict(np.zeros((1, SEQUENCE_LENGTH, FEATURES), dtype=np.float32), verbose=0)
print("Model loaded and warmed up. Gestures:", len(GESTURES))

# Keras models are not guaranteed thread-safe for concurrent inference.
# All calls to model.predict() must go through this lock — this matters
# specifically because CameraSession runs prediction on a background
# thread, and if more than one session is ever alive at once (see
# _camera_active_lock below), overlapping predict() calls on the same
# shared model instance can produce corrupted/nondeterministic output.
model_lock = threading.Lock()

# Only one physical camera exists. prediction.py never has this problem
# because it opens cv2.VideoCapture(0) exactly once for the entire program
# lifetime. main.py opens a new CameraSession (and a new VideoCapture) per
# websocket connection, so if the frontend ever reconnects while an old
# session's camera thread is still alive (page refresh, dev-mode double
# mount, a dropped/re-established socket), two threads can end up reading
# from — or both trying to open — the same camera device at once. On
# some backends (e.g. DirectShow on Windows) this doesn't cleanly error;
# it can silently interleave or serve stale frames from competing capture
# handles, which corrupts the 20-frame sliding window and produces
# confident-but-wrong predictions that have nothing to do with model
# quality. This lock ensures only one CameraSession can hold the camera
# at a time; a second session is rejected with an explicit error instead
# of silently fighting over the device.
_camera_active_lock = threading.Lock()

mp_holistic = mp.solutions.holistic

# ============================================================
# FEATURE EXTRACTION — byte-for-byte identical to prediction.py
# ============================================================

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    origin = coords[0]
    coords = coords - origin
    max_value = np.max(np.abs(coords))
    if max_value != 0:
        coords = coords / max_value
    return coords.flatten()


def extract_features(results):
    if results.pose_landmarks:
        pose = normalize_landmarks(results.pose_landmarks)
    else:
        pose = np.zeros(99, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = normalize_landmarks(results.left_hand_landmarks)
    else:
        left_hand = np.zeros(63, dtype=np.float32)

    if results.right_hand_landmarks:
        right_hand = normalize_landmarks(results.right_hand_landmarks)
    else:
        right_hand = np.zeros(63, dtype=np.float32)

    return np.concatenate([pose, left_hand, right_hand])


# ============================================================
# GROQ SENTENCE FORMATION
# ============================================================

def generate_sentence(words):
    if not words:
        return ""
    sign_sequence = " ".join(words)
    print("\n[generate_sentence] words:", sign_sequence)

    prompt = f"""
You are an English sentence-formation module for a continuous
sign language recognition system. Convert the recognized sign
words into ONE natural, grammatically correct English sentence.

Rules:
1. Preserve the intended meaning.
2. Do NOT invent new information.
3. You MAY add grammatical words (is, are, am, the, a, an, to, do, does, did, can, will).
4. You MAY correct verb forms when necessary.
5. Return ONLY ONE final sentence, no extra text.

Now process these recognized sign language words:

{sign_sequence}

Return ONLY the final grammatically correct sentence.
"""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You convert recognized sign-language words into one grammatically correct English sentence. Return only the final sentence."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=100,
        )
        sentence = response.choices[0].message.content.strip().strip('"')
        return sentence if sentence else sign_sequence
    except Exception as e:
        print("GROQ ERROR:", e)
        return sign_sequence


# ============================================================
# CAMERA SESSION — backend owns the webcam directly, exactly
# like prediction.py. Runs in a background thread (blocking
# cv2 calls don't belong on the async event loop); results are
# pushed to an asyncio.Queue via call_soon_threadsafe, and a
# separate async task drains that queue and sends over the
# websocket.
#
# IMPORTANT: only one CameraSession may hold the camera at a
# time (see _camera_active_lock). If start_camera() is called
# while another session already owns the camera, it fails
# loudly instead of opening a second competing capture handle.
# ============================================================

class CameraSession:
    def __init__(self, loop: asyncio.AbstractEventLoop, out_queue: asyncio.Queue):
        self.loop = loop
        self.out_queue = out_queue
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.cap = None

        # Single combined loop — deliberately matching prediction.py:
        # read frame, process, maybe predict, repeat, in strict order.
        # An earlier version of this file split capture and processing
        # into two threads connected by a maxsize=1 "latest frame only"
        # queue, to keep the video preview smooth even if recognition
        # lagged. That was a mistake: it silently drops frames whenever
        # processing can't keep up, which breaks the LSTM's assumption
        # that its 20-frame sequence window is 20 CONSECUTIVE frames at
        # a roughly constant time interval (the same assumption the
        # training data was recorded under). Dropped/irregularly-spaced
        # frames distort the apparent motion between frames in ways the
        # model never saw in training, producing confident-but-wrong
        # predictions. prediction.py proves a single sequential loop is
        # not inherently slow, so recognition correctness wins here —
        # if the preview needs to be smoother, throttle how often
        # MediaPipe/predict actually run (frame skipping) rather than
        # decoupling capture from processing.
        self.thread = None

        self.running = False
        self.detecting = False
        self.holds_camera_lock = False

        self.sequence = []
        self.prediction_history = []
        self.last_output = None
        self.sentence_words = []
        self.frame_counter = 0

    def _push(self, message: dict):
        self.loop.call_soon_threadsafe(self.out_queue.put_nowait, message)

    def start_camera(self):
        if self.cap is not None:
            return

        if not _camera_active_lock.acquire(blocking=False):
            print("[camera] Refused start_camera(): camera already in use by another session.")
            self._push({
                "type": "error",
                "message": "Camera is already in use by another active session. Close other tabs/connections and try again.",
            })
            return
        self.holds_camera_lock = True

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._push({"type": "error", "message": "Could not open camera on the server."})
            self.cap = None
            _camera_active_lock.release()
            self.holds_camera_lock = False
            return

        self.running = True
        self.detecting = True
        self.sequence = []
        self.prediction_history = []
        self.last_output = None
        self.frame_counter = 0

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def pause(self):
        self.detecting = False

    def resume(self):
        self.detecting = True

    def stop_camera(self):
        self.running = False
        self.detecting = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.holds_camera_lock:
            _camera_active_lock.release()
            self.holds_camera_lock = False

    def reset_words(self):
        self.sequence = []
        self.prediction_history = []
        self.last_output = None
        self.sentence_words = []
        self.frame_counter = 0

    def close(self):
        self.stop_camera()
        self.holistic.close()

    def _loop(self):
        """
        Byte-for-byte the same recognition logic as prediction.py's main
        loop — same variables, same strict stability check, no extra
        gating logic on top. Only additions vs. prediction.py: this runs
        per-session in a thread (prediction.py only ever has one instance
        total), pushes status/preview over a websocket instead of drawing
        on the frame with cv2.putText, and serializes model.predict()
        calls across sessions via model_lock.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            lighting_good = brightness >= MIN_BRIGHTNESS

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)

            left_detected = results.left_hand_landmarks is not None
            right_detected = results.right_hand_landmarks is not None
            hand_detected = left_detected or right_detected

            new_word = None

            if self.detecting and lighting_good and hand_detected:

                keypoints = extract_features(results)
                self.sequence.append(keypoints)

                if len(self.sequence) > SEQUENCE_LENGTH:
                    self.sequence.pop(0)

                if len(self.sequence) == SEQUENCE_LENGTH:

                    input_data = np.expand_dims(
                        np.array(self.sequence, dtype=np.float32),
                        axis=0,
                    )

                    # model_lock: serializes predict() calls across
                    # sessions. prediction.py doesn't need this
                    # because it only ever runs one instance; this
                    # backend can have multiple sessions, so this
                    # lock is the one addition here that isn't in
                    # prediction.py, purely for thread-safety.
                    with model_lock:
                        prediction = model.predict(input_data, verbose=0)

                    confidence = float(np.max(prediction))
                    predicted_class = int(np.argmax(prediction))

                    if predicted_class < len(GESTURES):

                        gesture = GESTURES[predicted_class]

                        if confidence >= THRESHOLD:

                            self.prediction_history.append(gesture)

                            if len(self.prediction_history) > HISTORY_LENGTH:
                                self.prediction_history.pop(0)

                            # Stability check — identical to prediction.py:
                            # requires HISTORY_LENGTH strictly consecutive
                            # identical predictions.
                            if (
                                len(self.prediction_history) == HISTORY_LENGTH
                                and self.prediction_history.count(gesture) == HISTORY_LENGTH
                            ):
                                if gesture != self.last_output:
                                    print("Detected Gesture:", gesture, " Confidence:", round(confidence, 3))

                                    self.sentence_words.append(gesture)
                                    self.last_output = gesture

                                    self.prediction_history = []
                                    new_word = gesture

            # Recognition status for this frame.
            self._push({
                "type": "status",
                "lighting_good": lighting_good,
                "hand_detected": hand_detected,
                "brightness": brightness,
                "words": self.sentence_words,
                "new_word": new_word,
                "detecting": self.detecting,
            })

            # Preview frame — every 3rd frame, purely for display, no
            # effect on recognition.
            self.frame_counter += 1
            if self.frame_counter % 3 == 0:
                small = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    preview_b64 = base64.b64encode(buf).decode("utf-8")
                    self._push({"type": "camera_frame", "data": f"data:image/jpeg;base64,{preview_b64}"})


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue = asyncio.Queue()
    session = CameraSession(loop, out_queue)

    async def sender():
        while True:
            message = await out_queue.get()
            try:
                await websocket.send_text(json.dumps(message))
            except Exception:
                break

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "start":
                session.start_camera()

            elif msg_type == "pause":
                session.pause()

            elif msg_type in ("stop",):
                session.pause()

            elif msg_type == "generate_sentence":
                session.pause()
                sentence = await asyncio.to_thread(generate_sentence, session.sentence_words)
                await websocket.send_text(json.dumps({"type": "sentence", "sentence": sentence}))
                session.stop_camera()

            elif msg_type == "back":
                if session.sentence_words:
                    session.sentence_words.pop()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "words": session.sentence_words,
                    "lighting_good": True,
                    "hand_detected": True,
                    "detecting": session.detecting,
                }))

            elif msg_type in ("start_over", "reset"):
                session.stop_camera()
                session.reset_words()
                await websocket.send_text(json.dumps({"type": "reset_ack"}))

    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        session.close()


# ============================================================
# TRANSLATE + SPEAK (unchanged)
# ============================================================

class TranslateRequest(BaseModel):
    text: str
    language: str

@app.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    translated = await asyncio.to_thread(translator.translate, req.text, req.language)
    return {"translated_text": translated}


class SpeakRequest(BaseModel):
    text: str
    lang: str = "en"

@app.post("/speak")
async def speak_endpoint(req: SpeakRequest):
    def generate_audio():
        tts = gTTS(text=req.text, lang=req.lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    audio_buf = await asyncio.to_thread(generate_audio)
    return StreamingResponse(audio_buf, media_type="audio/mpeg")


@app.get("/")
async def root():
    return {"status": "Signa backend running (server-owned camera mode)", "gestures": GESTURES}