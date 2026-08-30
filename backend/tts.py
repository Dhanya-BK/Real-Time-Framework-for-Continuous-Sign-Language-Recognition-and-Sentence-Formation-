from gtts import gTTS
from playsound import playsound
import os


class TextToSpeech:

    LANGUAGE_CODES = {
        "English": "en",
        "Kannada": "kn",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml"
    }

    def speak(self, text, language):

        lang = self.LANGUAGE_CODES.get(language, "en")

        filename = "speech.mp3"

        speech = gTTS(
            text=text,
            lang=lang,
            slow=False
        )

        speech.save(filename)

        playsound(filename)

        os.remove(filename)


        