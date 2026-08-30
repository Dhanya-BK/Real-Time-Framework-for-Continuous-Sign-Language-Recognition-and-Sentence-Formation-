from deep_translator import GoogleTranslator


class TranslatorModule:

    LANGUAGES = {
        "English": "en",
        "Kannada": "kn",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml"
    }

    def translate(self, text, language):

        if language == "English":
            return text

        lang_code = self.LANGUAGES.get(language)

        if lang_code is None:
            raise ValueError("Unsupported Language")

        translated = GoogleTranslator(
            source="auto",
            target=lang_code
        ).translate(text)

        return translated