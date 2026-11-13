<div align="center">

# 🤟 Signa — Where Signs Speak

### A Real-Time Framework for Continuous Sign Language Recognition and Sentence Formation

Webcam → MediaPipe landmarks → BiLSTM gesture model → LLM sentence formation → Translation → Text-to-Speech

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react&logoColor=black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00A98F)

</div>

---

## 📑 Table of Contents

[Overview](#-overview) · [Features](#-features) · [Architecture](#%EF%B8%8F-system-architecture)  · [Tech Stack](#-tech-stack) · [Project Structure](#-project-structure) · [Prerequisites](#-prerequisites) · [Getting Started](#-getting-started--step-by-step) · [Collect Data & Train](#-collect-your-own-data--train) · [API](#-api-reference)  · [Limitations](#%EF%B8%8F-limitations) · [Team](#-team)

---

## 📖 Overview

**Signa** is a real-time sign language recognition web app. It captures hand and body movements through a normal webcam, recognizes individual signs with a **Bidirectional LSTM (BiLSTM)** model, collects them into a word buffer, and uses an **LLM (Groq)** to turn the words into one grammatically correct sentence. The sentence can then be **translated** into Indian languages and **spoken aloud**.

It is built as a B.E. final-year major project (VTU, Dept. of AI & ML, VCET Puttur, 2026-27) to help bridge the communication gap between sign language users and people who don't know sign language — with **no gloves or special hardware**, just a webcam.

> Example: signs `HELLO` → `HOW` → `YOU` become **"Hello, how are you?"** → translated to Kannada / Hindi / Tamil / Telugu / Malayalam → spoken out loud.

---

## ✨ Features

- 🎥 **Real-time recognition** from a standard webcam (no special hardware)
- 🧠 **BiLSTM model** trained on 20-frame landmark sequences (225 features per frame)
- ✋ **MediaPipe Holistic** — pose + left hand + right hand landmarks
- 🛡️ **Stability filters** — confidence threshold, prediction history, repetition filter, motion gate
- 💡 **Quality checks** — live feedback for poor lighting / hand not detected
- 📝 **Sentence formation** using Groq LLMs, with automatic model fallbacks
- 🌐 **Translation** to English, Kannada, Hindi, Tamil, Telugu, Malayalam
- 🔊 **Text-to-speech** in the chosen language
- 🔁 Undo last word, pause/resume, start over
- 📊 Recognition logging to CSV (`backend/logs/recognition_log.csv`)
- 🧰 Includes tools to **collect your own dataset** and **train your own model**

---

## 🏗️ System Architecture

```
┌──────────┐   ┌──────────────────┐   ┌──────────────┐   ┌───────────────────┐
│  Webcam  │──▶│ MediaPipe        │──▶│ Normalize +  │──▶│ 20-frame sliding  │
│ (OpenCV) │   │ Holistic (225 f) │   │ motion check │   │ window buffer     │
└──────────┘   └──────────────────┘   └──────────────┘   └─────────┬─────────┘
                                                                    ▼
┌──────────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐
│ React UI     │◀──│ FastAPI     │◀──│ Groq LLM     │◀──│ BiLSTM classifier │
│ (Vite, WS)   │   │ WebSocket + │   │ sentence     │   │ + threshold +     │
│ translate/TTS│   │ REST        │   │ formation    │   │ history filter    │
└──────────────┘   └─────────────┘   └──────────────┘   └───────────────────┘
```

**Pipeline in short**

1. **Capture** – the backend opens the webcam with OpenCV.
2. **Landmarks** – MediaPipe Holistic gives 33 pose + 21 left-hand + 21 right-hand points (x, y, z) = **225 features/frame**.
3. **Normalize** – each part is shifted to its first landmark and scaled by its max absolute value (position/scale invariant).
4. **Sequence** – the last **20 frames** form one input window; windows with too little motion are skipped.
5. **Classify** – BiLSTM predicts the gesture; it is accepted only if confidence ≥ threshold **and** the same gesture repeats across the history window **and** differs from the last output.
6. **Sentence** – the word list is sent to Groq to produce a single clean sentence.
7. **Translate + Speak** – optional translation (Groq → Google → Ollama fallback) and TTS.

### Model

```
Input (20 × 225)
 → Bidirectional LSTM (64, return_sequences) → Dropout 0.3
 → Bidirectional LSTM (32)                    → Dropout 0.3
 → Dense 64 (ReLU)                            → Dropout 0.2
 → Dense N (Softmax)      # N = number of gestures in labels.py
```

Trained with an 80:20 stratified split, batch size 8, up to 50 epochs, early stopping (patience 7).

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Computer vision | OpenCV, MediaPipe Holistic |
| Deep learning | TensorFlow / Keras (BiLSTM), NumPy, scikit-learn |
| Backend | Python 3.10, FastAPI, WebSockets, Pydantic, Uvicorn |
| Sentence formation | Groq API (`openai/gpt-oss-120b`, with fallbacks) |
| Translation | Groq → `deep-translator` (Google) → Ollama (local Llama 3) |
| Text-to-speech | `gTTS` (backend `/speak`); `edge-tts` voices in `tts.py` |
| Frontend | React 18, Vite 5, lucide-react |

---

## 📁 Project Structure

```
Major_Project/
├── backend/
│   ├── main.py              # ✅ FastAPI server (WebSocket /ws, /translate, /speak) — USE THIS
│   ├── prediction.py        # Standalone desktop (OpenCV window) version of the pipeline
|   ├── dataset_collect.py   # Record your own gesture data via webcam
|   ├── train.py             # Train the BiLSTM model
│   ├── translator.py        # Translation module (Groq → Google → Ollama)
│   ├── tts.py               # Text-to-speech module (edge-tts → gTTS)
│   ├── labels.py            # List of gesture labels (order = model output order)
│   ├── gesture_model.h5     # Trained BiLSTM weights
│   ├── fix_model.py         # Patches .h5 batch_shape → batch_input_shape (Keras compat)
│   ├── test_groq.py         # Tests your Groq key / model access
│   ├── logs/                # recognition_log.csv (auto-created)
│   └── .env                 # 🔒 your secrets (NOT committed — see below)
├── signa-app/               # ✅ React + Vite frontend — USE THIS
│   ├── src/App.jsx
│   └── package.json
├── frontend/index.html      # Older single-file frontend (alternative)
├── data/                    # Collected landmark sequences: data/<gesture>/<n>.npy
```
## ✅ Prerequisites

| Requirement | Version | Why |
|---|---|---|
| **Python** | 3.10 (3.10.x recommended) | Backend, MediaPipe, TensorFlow |
| **Node.js + npm** | Node 18 or newer | React/Vite frontend |
| **Git** | any recent | Clone the repo |
| **Webcam** | built-in or USB | Sign capture |
| **Groq API key** | free | Sentence formation + translation |
| **Internet** | required | Groq, Google Translate, gTTS |
| **Hardware** | Intel i3+/4 GB RAM+ | No GPU needed for inference |

### Check what you already have

Open **PowerShell** and run:

```powershell
python --version     # should print Python 3.10.x
node --version       # should print v18 or higher
npm --version
git --version
```

If any command says *"not recognized"*, install it first:

- **Python 3.10** → https://www.python.org/downloads/release/python-3100/ (tick **"Add Python to PATH"** during install)
- **Node.js (LTS)** → https://nodejs.org
- **Git** → https://git-scm.com/downloads

> ⚠️ MediaPipe and TensorFlow can fail on Python 3.12/3.13. Stick to **3.10**.

---

## 🚀 Getting Started — Step by Step

> The commands are for **Windows PowerShell**. On Linux/macOS use `python3` and `source venv/bin/activate`.
> You will use **two terminals**: one for the backend, one for the frontend.


### Step 1 — Create a Python virtual environment (recommended)

A virtual environment keeps this project's packages separate from the rest of your system.

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2— Install backend dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
```text
tensorflow==2.15.0
mediapipe==0.10.14
opencv-python
numpy<2
fastapi
uvicorn[standard]
websockets
pydantic
python-dotenv
groq
gTTS
deep-translator
requests
edge-tts
scikit-learn
h5py
```

### Step 3— Get a Groq API key and create the `.env` file

1. Go to https://console.groq.com/keys and sign in (free).
2. Click **Create API Key** and copy it.
3. Inside the `backend/` folder, create a file named exactly **`.env`** (no `.txt` extension) with:

```env
GROQ_API_KEY=paste_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```
```powershell
python test_groq.py
```

### Step 4 — Start the backend server

From inside `backend/` with the virtual environment active:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait until you see:

```
Loading gesture recognition model...
Model loaded and warmed up. Gestures: <number>
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **Check:** open http://localhost:8000 in your browser — you should see a JSON message `"Signa backend running ..."` with the list of gestures.

**Leave this terminal running.**

---

### Step 5 — Install and start the frontend

Open a **new** PowerShell window (the backend keeps running in the first one):

```powershell
cd <your-repo-name>\signa-app
npm install
npm run dev
```

You'll see something like:

```
VITE v5.x  ready
➜  Local:   http://localhost:5173/
```

✅ **Check:** open **http://localhost:5173** — you should see the Signa splash screen, then the home screen.

---

### Step 6 — Use Signa

| # | Action | What you should see |
|---|---|---|
| 1 | Click **Start** | Backend opens the webcam; live preview appears |
| 2 | Check the status panel | **Lighting: Good** ✅ and **Hand Detected** ✅ |
| 3 | Perform a sign (e.g. *HELLO*), hold steady, then relax | Word appears in the sentence buffer |
| 4 | Continue signing (e.g. *HOW*, *YOU*) | Words are appended in order |
| 5 | Wrong word? Use **undo / back** | Last word is removed |
| 6 | Need a break? **Pause / Resume** | Detection stops/starts |
| 7 | Click **Generate Sentence** | Camera stops; LLM returns e.g. *"Hello, how are you?"* |
| 8 | Click **Translate** → pick a language → confirm | Sentence shown in Kannada/Hindi/Tamil/Telugu/Malayalam |
| 9 | Click **Speak** → pick a language | Sentence is spoken aloud |
| 10 | Click **Start Over** → confirm | Everything resets for a new sentence |

**Tips for good recognition**

- Sit in a bright, evenly lit room; avoid strong light behind you.
- Keep your upper body and both hands inside the frame.
- Sign at a steady pace and pause briefly between words.
- The same gesture is not added twice in a row — move to a different sign (or use Start Over) to repeat.
  
---

### Step 7 — Stop the project

- In each terminal press **Ctrl + C**.
- To leave the virtual environment: `deactivate`

### Next time you run it (quick start)

```powershell
# Terminal 1 — backend
cd <your-repo-name>\backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd <your-repo-name>\signa-app
npm run dev
```

Then open http://localhost:5173.

---

## 🧪 Collect Your Own Data & Train

### Collect gesture data

```powershell
# from the project root (folder containing dataset_collect.py and data/)
python dataset_collect.py
```

- Pick a gesture from the menu → press **S** to record a sequence (20 frames) → **Q** to quit.
- Each gesture records **30 sequences**, saved as `data/<gesture>/<n>.npy` with shape `(20, 225)`.
- Recording is blocked if lighting is too low or no hand is detected.
- To add a new sign, add its name to `backend/labels.py` (labels are imported by the scripts) and record data for it.

### Train the model

```powershell
python train.py
```

This loads all `.npy` sequences, trains the BiLSTM, prints the test accuracy, and saves **`gesture_model.h5`**. Copy it into `backend/` to use it.

> ⚠️ The **order of `GESTURES` in `labels.py` must match the order the model was trained with**. If you add/remove/reorder labels, retrain the model.

---

## 🔌 API Reference

| Type | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + list of supported gestures |
| WebSocket | `/ws` | Live control + streaming |
| POST | `/translate` | Body: `{"text": "...", "language": "Hindi"}` → `{"translated_text": "..."}` |
| POST | `/speak` | Body: `{"text": "...", "lang": "hi"}` → MP3 audio stream |

**WebSocket messages (client → server):** `start`, `pause`, `stop`, `generate_sentence`, `back`, `start_over` / `reset`
**Server → client:** `status` (lighting, hand, words, new word), `camera_frame` (JPEG preview), `sentence`, `reset_ack`, `error`

---

## ⚙️ Key Settings

| Setting | Web app (`main.py`) | Standalone (`prediction.py`) |
|---|---|---|
| Sequence length | 20 frames | 20 frames |
| Features / frame | 225 | 225 |
| Confidence threshold | 0.70 | 0.80 |
| Prediction history (stability) | 3 | 5 |
| Min. brightness | 50 | 50 |
| MediaPipe detection / tracking confidence | 0.7 / 0.7 | 0.7 / 0.7 |

Tune these constants at the top of the file to trade off speed vs. accuracy.

---

## 🗣️ Supported Signs

The vocabulary is defined in `backend/labels.py` (~100 everyday words), including:

**Greetings:** HELLO, Namaste, Welcome, Goodbye, Please, Sorry, Thank you
**Pronouns:** I, You, He, She, We, They, My, Your
**Actions:** Eat, Drink, Go, Come, Walk, Run, Sit, Sleep, Wake, Read, Write, Speak, Listen, See, Give, Take, Bring, Help, Learn, Work, Play, Wait, Stop, Start
**Questions:** What, Where, When, Who, Why, Which, How
**Objects/places:** Water, Milk, Tea, Coffee, Home, College, School, Book, Phone, Bag, Money, Clothes, Shoes, Medicine
**Descriptors/time:** Good, Bad, Big, Small, Hot, Cold, New, Old, Easy, Difficult, Fast, Slow, Today, Tomorrow, Yesterday, Morning, Evening, Night, Now, Later, Always, Never, Happy, Sad, Love, Like, Need, Want

---


## ⚠️ Limitations

- Recognizes a **fixed vocabulary of isolated words**, not full sign grammar or finger-spelling.
- Trained on a small, self-recorded dataset (30 sequences per sign) — accuracy depends on lighting, camera distance, and the signer.
- Needs internet for sentence formation, translation, and TTS.
- Non-manual cues (facial expressions) are not modeled yet.

## 🔮 Future Work

- Larger, multi-signer dataset and a bigger vocabulary
- Better robustness to lighting, occlusion, and background
- Offline translation and speech
- Android / iOS apps
- Facial-expression recognition and transformer-based models

---

## 👩‍💻 Team

Visvesvaraya Technological University, Belagavi — B.E. in Artificial Intelligence & Machine Learning
**Vivekananda College of Engineering & Technology, Puttur** · 2026-27

| Name | USN |
|---|---|
| Akshaya Shankari B | 4VP23AI004 |
| Dhanya B K | 4VP23AI016 |
| Sharanya Rai K | 4VP23AI045 |
| Srijanya S R | 4VP23AI053 |

**Guide:** Prof. Abhishek Kumar K, Assistant Professor, Dept. of AI & ML

---
