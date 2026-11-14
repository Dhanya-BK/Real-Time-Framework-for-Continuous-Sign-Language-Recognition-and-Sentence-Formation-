import os
import io
import re
import json
import base64
import asyncio
import threading
import time

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


import csv
from datetime import datetime, timezone

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "recognition_log.csv")


def _ensure_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp_utc", "session_id", "tester", "word", "confidence"]
            )


def log_recognized_word(session_id: str, tester: str, word: str, confidence: float):
    _ensure_log_file()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [datetime.now(timezone.utc).isoformat(), session_id, tester, word, round(float(confidence), 4)]
        )


_ensure_log_file()

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

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GROQ_MODEL_FALLBACKS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound-mini"]
groq_client = Groq(api_key=GROQ_API_KEY)

translator = TranslatorModule()

MODEL_PATH = "gesture_model.h5"
SEQUENCE_LENGTH = 20
FEATURES = 225
THRESHOLD = 0.70       
HISTORY_LENGTH = 3     

MIN_BRIGHTNESS = 50


MOTION_THRESHOLD = 0.005    
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7


BROADCAST_INTERVAL_SEC = 0.05  

DEBUG_FPS = True

PREVIEW_SCALE = 0.50         
PREVIEW_JPEG_QUALITY = 50  
PREVIEW_FPS = 10

LANGUAGE_CODES = {
    "English": "en", "Kannada": "kn", "Hindi": "hi",
    "Tamil": "ta", "Telugu": "te", "Malayalam": "ml",
}



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


model.predict(np.zeros((1, SEQUENCE_LENGTH, FEATURES), dtype=np.float32), verbose=0)
print("Model loaded and warmed up. Gestures:", len(GESTURES))

model_lock = threading.Lock()

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


SENTENCE_SYSTEM_PROMPT = (
    "You convert recognized sign-language words into one grammatically "
    "correct English sentence. Return only the final sentence."
)


def _build_sentence_prompt(sign_sequence: str) -> str:
    return f"""
You are an English sentence-formation module for a
continuous sign language recognition system.

The input contains English words recognized from
sign language gestures.

Convert the recognized sign words into ONE natural,
grammatically correct English sentence.

IMPORTANT RULES:

1. Preserve the intended meaning.

2. Do NOT invent new information.

3. Do NOT add new people, objects, places, actions,
emotions, or events.

4. You MAY add necessary grammatical words such as:
"is", "are", "am", "the", "a", "an", "to",
"do", "does", "did", "can", and "will".

5. You MAY correct verb forms when necessary.

6. If the signs form a question, return a grammatically
correct question.

7. If the signs form a greeting, preserve the greeting.

8. Return ONLY ONE final sentence.

9. Do NOT explain the correction.

10. Do NOT return extra text.

11. Use correct capitalization and punctuation.

12. Never return the input words unchanged. The output must
start with a capital letter and end with a full stop,
question mark or exclamation mark.

EXAMPLES:

Input:
HOW YOU

Output:
How are you?

Input:
HOW YOU DOING

Output:
How are you doing?

Input:
WHAT YOUR NAME

Output:
What is your name?

Input:
WHERE YOU GO

Output:
Where are you going?

Input:
I GO COLLEGE

Output:
I am going to college.

Input:
I EAT FOOD

Output:
I am eating food.

Input:
YOU EAT FOOD

Output:
You are eating food.

Input:
I WANT WATER

Output:
I want water.

Input:
TOMORROW I GO SCHOOL

Output:
I am going to school tomorrow.

Input:
HELLO HOW YOU

Output:
Hello, how are you?

Input:
THANK YOU

Output:
Thank you.

Input:
I SORRY

Output:
I am sorry.

Input:
HELLO

Output:
Hello.

Now process these recognized sign language words:

{sign_sequence}

Return ONLY the final grammatically correct sentence.

"""


def _clean_model_output(text: str) -> str:
    """Pull one clean sentence out of whatever the model returned."""
    text = (text or "").strip()
    if not text:
        return ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        lowered = line.lower()
        for prefix in ("final sentence:", "sentence:", "output:", "answer:"):
            if lowered.startswith(prefix):
                line = line[len(prefix):].strip()
                break

        line = line.strip().strip('"').strip("'").strip("`").strip()
        if line:
            return line

    return ""


def _is_well_formed(sentence: str, sign_sequence: str) -> bool:
    """A usable sentence starts with a capital, ends with . ? or !, and
    is not just the input words echoed back."""
    if not sentence:
        return False
    if sentence.strip().upper() == sign_sequence.strip().upper() and sentence == sentence.upper():
        return False

    first_letter = next((c for c in sentence if c.isalpha()), "")
    if not first_letter or not first_letter.isupper():
        return False

    return sentence[-1] in ".?!"


def generate_sentence(words):
    if not words:
        return ""

    # Same style as the prompt examples (uppercase words).
    clean_words = [str(w).strip().upper() for w in words if str(w).strip()]
    if not clean_words:
        return ""

    sign_sequence = " ".join(clean_words)
    print("\n[generate_sentence] words:", sign_sequence)

    prompt = _build_sentence_prompt(sign_sequence)

    models_to_try = [GROQ_MODEL] + [m for m in GROQ_MODEL_FALLBACKS if m != GROQ_MODEL]
    last_error = None
    best_candidate = ""

    for model_name in models_to_try:
        try:
            kwargs = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": SENTENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1000,
            )
            
            if "gpt-oss" in model_name:
                kwargs["reasoning_effort"] = "low"

            response = groq_client.chat.completions.create(**kwargs)

            raw = response.choices[0].message.content
            sentence = _clean_model_output(raw)

            if not sentence:
                print(f"[generate_sentence] WARNING: {model_name} returned empty content, trying next fallback.")
                continue

            if _is_well_formed(sentence, sign_sequence):
                print(f"[generate_sentence] Groq output ({model_name}):", sentence)
                return sentence

            print(f"[generate_sentence] {model_name} gave a badly formed sentence: {sentence!r}. Trying next fallback.")
            best_candidate = best_candidate or sentence

        except Exception as e:
            last_error = e
            print(f"[generate_sentence] Model '{model_name}' failed: {e}")
            continue

    print("[generate_sentence] WARNING: no model gave a well-formed sentence.")
    if last_error is not None:
        print("[generate_sentence] Last Groq error:", last_error)
        print("[generate_sentence] Check GROQ_API_KEY, internet access, and whether the models are still available.")

    fallback = best_candidate or " ".join(clean_words).lower()
    fallback = fallback.strip()
    if fallback:
        fallback = fallback[0].upper() + fallback[1:]
        if fallback[-1] not in ".?!":
            fallback += "."
    return fallback



class CameraSession:
    def __init__(self, loop: asyncio.AbstractEventLoop, out_queue: asyncio.Queue ,tester:str="unknown"):
        self.loop = loop
        self.out_queue = out_queue
        self.session_id = f"{int(datetime.now(timezone.utc).timestamp())}-{id(self) % 10000}"
        self.tester = tester or "unknown"
        
        self.holistic = None
        self.cap = None

        
        self.thread = None

        self.running = False
        self.detecting = False
        self.holds_camera_lock = False

        self.sequence = []
        self.prediction_history = []
        self.last_output = None
        self.sentence_words = []


        self.latest_status = None
        self.latest_preview = None

        
        self.latest_raw_frame = None
        self.preview_thread = None

    def _push(self, message: dict):
        """Immediate, out-of-band push — used only for things that must
        not wait for the next broadcaster tick (e.g. a newly recognized
        word, or an error)."""
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

        if self.holistic is not None:
            self.holistic.close()
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        self.running = True
        self.detecting = True
        self.sequence = []
        self.prediction_history = []
        self.last_output = None
        self.latest_status = None
        self.latest_preview = None

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()

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
        if self.preview_thread:
            self.preview_thread.join(timeout=2)
            self.preview_thread = None
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

    def close(self):
        self.stop_camera()
        if self.holistic is not None:
            self.holistic.close()
            self.holistic = None

    def _loop(self):
        """
        Byte-for-byte matches prediction.py's recognition logic now: every
        frame gets holistic.process() (no frame-skipping), and the same
        MOTION_THRESHOLD gate prediction.py uses decides whether to run
        the model at all for that window. The only things added beyond
        prediction.py are: (1) model_lock, since this backend can run more
        than one session against the same shared model instance, and
        (2) storing status/raw-frame into self.latest_* instead of calling
        cv2.imshow(), since the frame has to reach a browser instead of a
        local window.
        """
        fps_count, fps_start = 0, time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if DEBUG_FPS:
                fps_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 2:
                    print(f"[fps] {fps_count / elapsed:.1f}")
                    fps_count, fps_start = 0, time.time()

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

                    seq_array = np.array(self.sequence, dtype=np.float32)

                   
                    frame_diffs = np.diff(seq_array, axis=0)
                    motion = float(np.mean(np.abs(frame_diffs)))

                    if motion >= MOTION_THRESHOLD:

                        input_data = np.expand_dims(seq_array, axis=0)

                       
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

                                
                                if (
                                    len(self.prediction_history) == HISTORY_LENGTH
                                    and self.prediction_history.count(gesture) == HISTORY_LENGTH
                                ):
                                    if gesture != self.last_output:
                                        print("Detected Gesture:", gesture, " Confidence:", round(confidence, 3))

                                        self.sentence_words.append(gesture)
                                        self.last_output = gesture

                                        self.prediction_history = []
                                        self.sequence = []
                                        new_word = gesture
                                        log_recognized_word(self.session_id, self.tester, gesture, confidence)

            status_message = {
                "type": "status",
                "lighting_good": lighting_good,
                "hand_detected": hand_detected,
                "brightness": brightness,
                "words": self.sentence_words,
                "new_word": new_word,
                "detecting": self.detecting,
            }
            self.latest_status = status_message

            
            if new_word is not None:
                self._push(status_message)

            
            self.latest_raw_frame = frame

    def _preview_loop(self):
        """Runs on its own thread, fully decoupled from cap.read(). Picks
        up whatever frame _loop most recently stashed in
        self.latest_raw_frame, resizes + JPEG-encodes + base64-encodes it,
        and stores the result in self.latest_preview for the websocket
        broadcaster to pick up. Runs at a fixed PREVIEW_FPS regardless of how
        fast the camera/recognition loop is running, so this thread can
        never be the reason cap.read() gets delayed."""
        last_frame_id = None
        while self.running:
            frame = self.latest_raw_frame
            if frame is not None and id(frame) != last_frame_id:
                last_frame_id = id(frame)
                small = cv2.resize(frame, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
                if ok:
                    preview_b64 = base64.b64encode(buf).decode("utf-8")
                    self.latest_preview = f"data:image/jpeg;base64,{preview_b64}"
            time.sleep(1 / PREVIEW_FPS)


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
        """Sends immediate/out-of-band messages (errors, new-word status,
        sentence results, reset acks) as soon as they're queued."""
        while True:
            message = await out_queue.get()
            try:
                await websocket.send_text(json.dumps(message))
            except Exception:
                break

    async def broadcaster():
        """FIX (camera stutter): sends only the latest status/preview
        snapshot, at a fixed rate, instead of the camera thread pushing
        one message per frame into an unbounded queue. This is what
        actually stops the backlog-then-burst behavior — a slow client
        just gets fewer updates, never a growing queue of stale ones."""
        last_status_sent = None
        last_preview_sent = None
        while True:
            await asyncio.sleep(BROADCAST_INTERVAL_SEC)
            status = session.latest_status
            preview = session.latest_preview
            try:
                if status is not None and status is not last_status_sent:
                    await websocket.send_text(json.dumps(status))
                    last_status_sent = status
                if preview is not None and preview is not last_preview_sent:
                    await websocket.send_text(json.dumps({"type": "camera_frame", "data": preview}))
                    last_preview_sent = preview
            except Exception:
                break

    sender_task = asyncio.create_task(sender())
    broadcaster_task = asyncio.create_task(broadcaster())

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "start":
                tester = msg.get("tester")
                if tester:
                    session.tester = tester
                session.start_camera()

            elif msg_type == "pause":
                session.pause()

            elif msg_type in ("stop",):
                session.pause()

            elif msg_type == "generate_sentence":
                session.pause()
                sentence = await asyncio.to_thread(generate_sentence, list(session.sentence_words))
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
        broadcaster_task.cancel()
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
