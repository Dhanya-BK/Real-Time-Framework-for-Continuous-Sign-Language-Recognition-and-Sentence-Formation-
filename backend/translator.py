"""
translator.py
Translation for Signa: English -> Kannada / Hindi / Tamil / Telugu / Malayalam.

Backends are tried in the order set in BACKEND_ORDER:
    "groq"   -> Groq cloud (openai/gpt-oss-120b). Needs GROQ_API_KEY in .env
    "google" -> Google Translate through deep_translator (free, but rate limited)
    "ollama" -> local Llama 3 (no internet, but weaker on Indian languages)

If one backend fails (rate limit, no internet, wrong language output),
the next one is tried automatically. Results are cached, so the same
sentence + language is never translated twice.

The interface is unchanged:
    TranslatorModule().translate(text, language)
"""

import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from deep_translator import GoogleTranslator


# ============================================================
# SETTINGS
# ============================================================

BACKEND_ORDER = ["groq", "google", "ollama"]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TIMEOUT = 30

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 60


class TranslatorModule:

    LANGUAGES = {
        "English": "en",
        "Kannada": "kn",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml"
    }

    # Unicode ranges, used to check that the reply really is in the target script
    SCRIPT_RANGES = {
        "Kannada": (0x0C80, 0x0CFF),
        "Hindi": (0x0900, 0x097F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Malayalam": (0x0D00, 0x0D7F),
    }

    def __init__(self):
        self.cache = {}

    # --------------------------------------------------------
    # PUBLIC
    # --------------------------------------------------------
    def translate(self, text, language):

        if language == "English":
            return text

        lang_code = self.LANGUAGES.get(language)

        if lang_code is None:
            raise ValueError("Unsupported Language")

        key = (text, language)
        if key in self.cache:
            return self.cache[key]

        errors = []

        for backend in BACKEND_ORDER:
            try:
                if backend == "groq":
                    result = self._translate_groq(text, language)
                elif backend == "google":
                    result = self._translate_google(text, lang_code)
                elif backend == "ollama":
                    result = self._translate_ollama(text, language)
                else:
                    continue

                result = (result or "").strip().strip('"').strip()

                if not self._is_valid(result, language):
                    raise ValueError("reply is not in the target language")

                print(f"[translator] used: {backend}")
                self.cache[key] = result
                return result

            except Exception as error:
                errors.append(f"{backend}: {error}")
                print(f"[translator] {backend} failed -> {error}")

        raise RuntimeError("All translation backends failed. " + " | ".join(errors))

    # --------------------------------------------------------
    # CHECK THAT THE OUTPUT USES THE TARGET SCRIPT
    # --------------------------------------------------------
    def _is_valid(self, text, language):
        if not text:
            return False
        low, high = self.SCRIPT_RANGES[language]
        return any(low <= ord(ch) <= high for ch in text)

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------
    @staticmethod
    def _prompt(text, language):
        return (
            f"Translate this English sentence into {language}. "
            f"Use the {language} script. "
            "Return ONLY the translated sentence, with no explanation, "
            "no transliteration and no quotation marks.\n\n"
            f"Sentence: {text}"
        )

    # --------------------------------------------------------
    # BACKEND 1: GROQ
    # --------------------------------------------------------
    def _translate_groq(self, text, language):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise translator. Return only the translation."
                    },
                    {"role": "user", "content": self._prompt(text, language)},
                ],
                "temperature": 0,
                "reasoning_effort": "low",
                "max_completion_tokens": 1000,
            },
            timeout=GROQ_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # --------------------------------------------------------
    # BACKEND 2: GOOGLE (deep_translator)
    # --------------------------------------------------------
    def _translate_google(self, text, lang_code):
        return GoogleTranslator(source="auto", target=lang_code).translate(text)

    # --------------------------------------------------------
    # BACKEND 3: OLLAMA (local)
    # --------------------------------------------------------
    def _translate_ollama(self, text, language):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": self._prompt(text, language)}
                ],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 200},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        reply = response.json()["message"]["content"].strip()
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        return lines[0] if lines else ""
