# ============================================================
# CONTINUOUS SIGN LANGUAGE RECOGNITION
# GROQ + TRANSLATION + TEXT-TO-SPEECH
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import cv2
import numpy as np
import mediapipe as mp

from dotenv import load_dotenv

from groq import Groq

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

from labels import GESTURES

# Import Translation and TTS modules
from translator import TranslatorModule
from tts import TextToSpeech


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GROQ SETTINGS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "\nGROQ_API_KEY was not found.\n"
        "Create a .env file in the project folder.\n"
        "Add:\n"
        "GROQ_API_KEY=your_actual_groq_api_key\n"
    )

# Groq model
GROQ_MODEL = "openai/gpt-oss-120b"

# Create Groq client
client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# INITIALIZE TRANSLATOR AND TTS
# ============================================================

translator = TranslatorModule()
tts = TextToSpeech()


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "gesture_model.h5"

SEQUENCE_LENGTH = 20

FEATURES = 225

# Minimum model prediction confidence
THRESHOLD = 0.80

# Number of same predictions required for stability
HISTORY_LENGTH = 5

# Minimum acceptable brightness
MIN_BRIGHTNESS = 50

# MediaPipe confidence
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Minimum average frame-to-frame movement (in normalized landmark units)
# required before we even attempt a prediction. Below this, the hand is
# considered "resting" or "transitioning" between signs, not actively
# signing, so we skip prediction entirely rather than let the model
# guess on a static/blend window. Tune this using the printed [motion]
# values: watch it while resting vs. while actually signing, then pick
# a threshold between the two ranges.
MOTION_THRESHOLD = 0.015


# ============================================================
# LOAD TRAINED LSTM MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING GESTURE RECOGNITION MODEL")
print("=" * 60)

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQUENCE_LENGTH, FEATURES)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(GESTURES), activation="softmax"))

model.load_weights(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# DISPLAY AVAILABLE GESTURES
# ============================================================

print()
print("=" * 60)
print("AVAILABLE GESTURES")
print("=" * 60)

for index, gesture in enumerate(GESTURES, start=1):
    print(index, ":", gesture)


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(landmarks):
    coords = []

    # Extract x, y, z coordinates
    for landmark in landmarks.landmark:
        coords.append([landmark.x, landmark.y, landmark.z])

    coords = np.array(coords, dtype=np.float32)

    # Use first landmark as origin
    origin = coords[0]
    coords = coords - origin

    # Scale normalization
    max_value = np.max(np.abs(coords))

    if max_value != 0:
        coords = coords / max_value

    return coords.flatten()


# ============================================================
# EXTRACT 225 FEATURES
# ============================================================

def extract_features(results):
    # --------------------------------------------------------
    # POSE
    #
    # 33 landmarks x 3 = 99
    # --------------------------------------------------------
    if results.pose_landmarks:
        pose = normalize_landmarks(results.pose_landmarks)
    else:
        pose = np.zeros(99, dtype=np.float32)

    # --------------------------------------------------------
    # LEFT HAND
    #
    # 21 landmarks x 3 = 63
    # --------------------------------------------------------
    if results.left_hand_landmarks:
        left_hand = normalize_landmarks(results.left_hand_landmarks)
    else:
        left_hand = np.zeros(63, dtype=np.float32)

    # --------------------------------------------------------
    # RIGHT HAND
    #
    # 21 landmarks x 3 = 63
    # --------------------------------------------------------
    if results.right_hand_landmarks:
        right_hand = normalize_landmarks(results.right_hand_landmarks)
    else:
        right_hand = np.zeros(63, dtype=np.float32)

    # --------------------------------------------------------
    # COMBINE
    #
    # Pose      = 99
    # Left hand = 63
    # Right     = 63
    #
    # Total = 225
    # --------------------------------------------------------
    features = np.concatenate([pose, left_hand, right_hand])

    return features


# ============================================================
# GROQ GRAMMAR AND SENTENCE FORMATION
# ============================================================

def generate_sentence(words):
    # --------------------------------------------------------
    # Check whether recognized words exist
    # --------------------------------------------------------
    if not words:
        return ""

    # --------------------------------------------------------
    # Convert recognized signs to text
    # --------------------------------------------------------
    sign_sequence = " ".join(words)

    print()
    print("=" * 60)
    print("SENDING SIGNS TO GROQ")
    print("=" * 60)
    print("Recognized signs:", sign_sequence)

    # ========================================================
    # PROMPT
    # ========================================================
    prompt = f"""
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

Now process these recognized sign language words:

{sign_sequence}

Return ONLY the final grammatically correct sentence.

"""

    # ========================================================
    # CALL GROQ
    # ========================================================
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert recognized sign-language "
                        "words into one grammatically correct "
                        "English sentence. Return only the "
                        "final sentence."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=100
        )

        # ----------------------------------------------------
        # GET RESPONSE
        # ----------------------------------------------------
        sentence = response.choices[0].message.content.strip()

        # ----------------------------------------------------
        # REMOVE QUOTATION MARKS
        # ----------------------------------------------------
        sentence = sentence.strip('"')

        # ----------------------------------------------------
        # CHECK EMPTY RESPONSE
        # ----------------------------------------------------
        if not sentence:
            print("Groq returned an empty response.")
            return sign_sequence

        # ----------------------------------------------------
        # DISPLAY FINAL SENTENCE
        # ----------------------------------------------------
        print()
        print("=" * 60)
        print("GRAMMATICALLY CORRECT ENGLISH SENTENCE")
        print("=" * 60)
        print(sentence)
        print("=" * 60)

        return sentence

    # ========================================================
    # GROQ ERROR
    # ========================================================
    except Exception as error:
        print()
        print("=" * 60)
        print("GROQ ERROR")
        print("=" * 60)
        print(str(error))
        print()
        print("Check your internet connection.")
        print("Check the API key in the .env file.")
        print("Check whether the selected Groq model is available.")
        print("=" * 60)

        # Return raw recognized signs if Groq fails
        return sign_sequence


# ============================================================
# LANGUAGE SELECTION
# ============================================================

def select_language():
    print()
    print("=" * 50)
    print("SELECT LANGUAGE")
    print("=" * 50)
    print("1. English")
    print("2. Kannada")
    print("3. Hindi")
    print("4. Tamil")
    print("5. Telugu")
    print("6. Malayalam")

    language_map = {
        "1": "English",
        "2": "Kannada",
        "3": "Hindi",
        "4": "Tamil",
        "5": "Telugu",
        "6": "Malayalam"
    }

    choice = input("\nEnter language number: ").strip()

    language = language_map.get(choice)

    if language is None:
        print("\nInvalid language selected.")
        return None

    return language


# ============================================================
# TRANSLATE / SPEAK
# ============================================================

def process_sentence(sentence):
    if not sentence:
        print("\nNo sentence available.")
        return

    print()
    print("=" * 60)
    print("FINAL RECOGNIZED SENTENCE")
    print("=" * 60)
    print(sentence)
    print("=" * 60)

    print()
    print("Choose Operation")
    print("1. Translate")
    print("2. Speak")
    print("3. Translate + Speak")

    choice = input("\nEnter your choice: ").strip()

    if choice not in ["1", "2", "3"]:
        print("\nInvalid operation.")
        return

    language = select_language()

    if language is None:
        return

    translated_text = sentence

    # ========================================================
    # TRANSLATE
    # ========================================================
    if language != "English":
        try:
            translated_text = translator.translate(sentence, language)
        except Exception as error:
            print()
            print("TRANSLATION ERROR")
            print(str(error))
            return

    # ========================================================
    # TRANSLATE ONLY
    # ========================================================
    if choice == "1":
        print()
        print(f"TRANSLATION - {language.upper()}")
        print(translated_text)

    # ========================================================
    # SPEAK ONLY
    # ========================================================
    elif choice == "2":
        try:
            tts.speak(translated_text, language)
            print("\nSpeech completed.")
        except Exception as error:
            print("\nTEXT-TO-SPEECH ERROR")
            print(str(error))

    # ========================================================
    # TRANSLATE + SPEAK
    # ========================================================
    elif choice == "3":
        print()
        print(f"TRANSLATION - {language.upper()}")
        print(translated_text)

        try:
            tts.speak(translated_text, language)
            print("\nTranslation and speech completed.")
        except Exception as error:
            print("\nTEXT-TO-SPEECH ERROR")
            print(str(error))


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    holistic.close()
    exit()


# ============================================================
# VARIABLES
# ============================================================

sequence = []
prediction_history = []
last_output = None
sentence_words = []
final_sentence = ""
detecting = False


# ============================================================
# START MESSAGE
# ============================================================

print()
print("=" * 60)
print("CONTINUOUS SIGN LANGUAGE RECOGNITION")
print("=" * 60)
print()
print("Controls:")
print("S -> Start / New Recognition")
print("F -> Generate Sentence")
print("C -> Clear Recognized Signs")
print("Q -> Quit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ CAMERA
    # ========================================================
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        continue

    # Flip camera
    frame = cv2.flip(frame, 1)

    # ========================================================
    # LIGHTING CHECK
    # ========================================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    lighting_good = (brightness >= MIN_BRIGHTNESS)

    # ========================================================
    # MEDIAPIPE
    # ========================================================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    # ========================================================
    # HAND DETECTION
    # ========================================================
    left_detected = (results.left_hand_landmarks is not None)

    right_detected = (results.right_hand_landmarks is not None)

    hand_detected = (left_detected or right_detected)

    # ========================================================
    # DISPLAY STATUS
    # ========================================================
    cv2.putText(
        frame,
        f"Brightness: {int(brightness)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    lighting_text = "Lighting: GOOD" if lighting_good else "Lighting: TOO LOW"

    lighting_color = (0, 255, 0) if lighting_good else (0, 0, 255)

    cv2.putText(
        frame,
        lighting_text,
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        lighting_color,
        2
    )

    hand_text = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED"

    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)

    cv2.putText(
        frame,
        hand_text,
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        hand_color,
        2
    )

    # ========================================================
    # CONTINUOUS RECOGNITION
    # ========================================================
    if detecting and lighting_good and hand_detected:

        keypoints = extract_features(results)

        sequence.append(keypoints)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # ================================================
        # PREDICTION
        # ================================================
        if len(sequence) == SEQUENCE_LENGTH:

            seq_array = np.array(sequence, dtype=np.float32)

            # --------------------------------------------
            # MOVEMENT GATE
            #
            # Skip prediction entirely if the hand is
            # basically static (resting between signs, or
            # in a slow transition) instead of letting the
            # model confidently guess on a non-gesture
            # window.
            # --------------------------------------------
            frame_diffs = np.diff(seq_array, axis=0)
            motion = float(np.mean(np.abs(frame_diffs)))
            print(f"[motion] {motion:.4f}")

            if motion < MOTION_THRESHOLD:
                pass
            else:

                input_data = np.expand_dims(seq_array, axis=0)

                prediction = model.predict(input_data, verbose=0)

                confidence = np.max(prediction)

                predicted_class = np.argmax(prediction)

                if predicted_class < len(GESTURES):

                    gesture = GESTURES[predicted_class]

                    if confidence >= THRESHOLD:

                        prediction_history.append(gesture)

                        if len(prediction_history) > HISTORY_LENGTH:
                            prediction_history.pop(0)

                        # =================================================
                        # STABILITY CHECK
                        # =================================================
                        if (
                            len(prediction_history) == HISTORY_LENGTH
                            and prediction_history.count(gesture) == HISTORY_LENGTH
                        ):

                            # Duplicate filter
                            if gesture != last_output:

                                print()
                                print("Detected Gesture:", gesture)
                                print("Confidence:", round(float(confidence), 3))

                                sentence_words.append(gesture)

                                last_output = gesture

                                prediction_history = []
                                sequence = []

    # ========================================================
    # DISPLAY RECOGNIZED SIGNS
    # ========================================================
    words_text = " ".join(sentence_words)

    cv2.putText(
        frame,
        "Signs: " + words_text[:60],
        (30, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    # ========================================================
    # DISPLAY FINAL SENTENCE
    # ========================================================
    if final_sentence:

        display_sentence = final_sentence[:70]

        cv2.putText(
            frame,
            "Sentence:",
            (30, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            display_sentence,
            (30, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    # ========================================================
    # DETECTION STATUS
    # ========================================================
    if detecting:
        status = "STATUS: DETECTING"
        status_color = (0, 255, 0)
    else:
        status = "STATUS: PAUSED"
        status_color = (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (30, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    # ========================================================
    # CONTROLS
    # ========================================================
    cv2.putText(
        frame,
        "S: Start | F: Sentence | C: Clear | Q: Quit",
        (30, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SHOW CAMERA
    # ========================================================
    cv2.imshow("Continuous Sign Language Recognition", frame)

    # ========================================================
    # KEYBOARD INPUT
    # ========================================================
    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # START
    # ========================================================
    if key == ord("s"):

        detecting = True
        sequence = []
        prediction_history = []
        last_output = None
        sentence_words = []
        final_sentence = ""

        print()
        print("RECOGNITION STARTED")
        print("Perform your signs.")
        print("Press F when finished.")

    # ========================================================
    # GENERATE SENTENCE
    # ========================================================
    elif key == ord("f"):

        if sentence_words:

            detecting = False

            print()
            print("Recognized sign sequence:")
            print(sentence_words)

            final_sentence = generate_sentence(sentence_words)

            if final_sentence:
                process_sentence(final_sentence)

        else:
            print("\nNo gestures have been recognized yet.")

    # ========================================================
    # CLEAR
    # ========================================================
    elif key == ord("c"):

        sequence = []
        prediction_history = []
        last_output = None
        sentence_words = []
        final_sentence = ""

        print("\nRecognized signs cleared.")

    # ========================================================
    # QUIT
    # ========================================================
    elif key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
holistic.close()

print()
print("=" * 60)
print("PROGRAM CLOSED")
print("=" * 60)