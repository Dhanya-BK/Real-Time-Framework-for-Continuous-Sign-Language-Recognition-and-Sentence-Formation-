"""
tts.py
Text-to-speech for Signa: English, Kannada, Hindi, Tamil, Telugu, Malayalam.

Primary : edge-tts (Microsoft neural voices, free, no API key, needs internet)
Fallback: gTTS     (Google, free, needs internet)

The interface is unchanged:
    TextToSpeech().speak(text, language)

Install:  pip install edge-tts gTTS playsound==1.2.2
Check voices:  edge-tts --list-voices   (look for en-IN, hi-IN, kn-IN, ta-IN, te-IN, ml-IN)
"""

import asyncio
import os
import platform
import uuid


class TextToSpeech:

    # edge-tts voices (one per language). Change here if a voice name is not found.
    EDGE_VOICES = {
        "English": "en-IN-NeerjaNeural",
        "Kannada": "kn-IN-SapnaNeural",
        "Hindi": "hi-IN-SwaraNeural",
        "Tamil": "ta-IN-PallaviNeural",
        "Telugu": "te-IN-ShrutiNeural",
        "Malayalam": "ml-IN-SobhanaNeural",
    }

    # gTTS language codes (fallback)
    LANGUAGE_CODES = {
        "English": "en",
        "Kannada": "kn",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml"
    }

    # --------------------------------------------------------
    # PUBLIC
    # --------------------------------------------------------
    def speak(self, text, language):

        if not text:
            return

        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"

        # 1) edge-tts, 2) gTTS
        try:
            self._save_edge(text, language, filename)
            print("[tts] used: edge-tts")
        except Exception as error:
            print(f"[tts] edge-tts failed -> {error}")
            try:
                self._save_gtts(text, language, filename)
                print("[tts] used: gTTS")
            except Exception as error2:
                raise RuntimeError(
                    f"All text-to-speech backends failed. edge-tts: {error} | gTTS: {error2}"
                )

        self._play(filename)

    # --------------------------------------------------------
    # BACKEND 1: EDGE-TTS
    # --------------------------------------------------------
    def _save_edge(self, text, language, filename):
        import edge_tts

        voice = self.EDGE_VOICES.get(language, self.EDGE_VOICES["English"])

        async def run():
            await edge_tts.Communicate(text, voice).save(filename)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            raise RuntimeError("edge-tts produced an empty file")

    # --------------------------------------------------------
    # BACKEND 2: GTTS
    # --------------------------------------------------------
    def _save_gtts(self, text, language, filename):
        from gtts import gTTS

        lang = self.LANGUAGE_CODES.get(language, "en")
        gTTS(text=text, lang=lang, slow=False).save(filename)

    # --------------------------------------------------------
    # PLAY AUDIO
    # --------------------------------------------------------
    def _play(self, filename):
        try:
            from playsound import playsound
            playsound(filename)
            os.remove(filename)
            return
        except Exception as error:
            print(f"[tts] playsound failed -> {error}")

        # Fallback: open with the system player (file is kept so you can play it)
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(filename)
            elif system == "Darwin":
                os.system(f'afplay "{filename}"')
            else:
                os.system(f'mpg123 "{filename}" 2>/dev/null || xdg-open "{filename}"')
            print(f"[tts] audio saved as {filename}")
        except Exception as error:
            print(f"[tts] could not play audio -> {error}")
            print(f"[tts] audio saved as {filename}")
