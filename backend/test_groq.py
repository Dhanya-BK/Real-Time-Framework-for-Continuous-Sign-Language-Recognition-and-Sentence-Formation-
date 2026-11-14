"""
Standalone Groq sentence-formation test.
Run this directly to see the REAL error if the API call is failing,
independent of the FastAPI/WebSocket server.

Usage (from backend\\ folder, in PowerShell):
    python test_groq.py
"""
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("Key loaded:", bool(GROQ_API_KEY), "| length:", len(GROQ_API_KEY) if GROQ_API_KEY else 0)

from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY)

print("\n--- Available models for your API key ---")
try:
    models = groq_client.models.list()
    for m in models.data:
        print(" -", m.id)
except Exception:
    import traceback
    print("Could not list models:")
    traceback.print_exc()
print("--- end of model list ---\n")

GROQ_MODEL = "openai/gpt-oss-120b"

test_word_lists = [
    ["HELLO", "how", "you"],
    ["I", "go", "school"],
    ["what", "your", "name"],
]

prompt_template = """You are an English sentence-formation module for a continuous
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

Return ONLY the final grammatically correct sentence."""

for words in test_word_lists:
    sign_sequence = " ".join(words)
    print("\n" + "=" * 50)
    print("INPUT:", sign_sequence)
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You convert recognized sign-language words into one grammatically correct English sentence. Return only the final sentence."},
                {"role": "user", "content": prompt_template.format(sign_sequence=sign_sequence)},
            ],
            temperature=0,
            max_tokens=100,
        )
        sentence = response.choices[0].message.content.strip().strip('"')
        print("OUTPUT:", sentence)
    except Exception:
        print("!!! GROQ CALL FAILED !!!")
        traceback.print_exc()