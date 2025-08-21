import re
from .ai_models import whisper_model, sentence_model
from deep_translator import GoogleTranslator
from .vektorizer.text_vectorizer import TextVectorizer
import pytesseract
from PIL import Image
text_vectorizer = TextVectorizer()

class ProcessingLogic:
    def __init__(self, whisper_model, sentence_model):
        self.whisper_model = whisper_model
        self.sentence_model = sentence_model

    def transcribe_audio(self, audio_path: str) -> str:
        print(f"Transkribiere Audio: {audio_path}...")
        segments, _ = self.whisper_model.transcribe(audio_path, beam_size=5)
        full_text = " ".join(segment.text for segment in segments)
        return full_text

    def extract_abstract_and_keywords(self, text_content: str) -> dict:
        abstract = None
        keywords = []
        abstract_match = re.search(r'abstract\n(.*?)\n\n', text_content, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        keywords_match = re.search(r'keywords:\s*(.*)', text_content, re.IGNORECASE)
        if keywords_match:
            keywords = [k.strip() for k in keywords_match.group(1).split(',') if k.strip()]
        return {'abstract': abstract, 'keywords': keywords}

    def process_text_for_ai(self, text_content: str):
        embeddings = self.sentence_model.encode(text_content).tolist()
        tags = []
        if "klimawandel" in text_content.lower() or "climate change" in text_content.lower():
            tags.append("SDG 13")
        if "armut" in text_content.lower() or "poverty" in text_content.lower():
            tags.append("SDG 1")

        extracted_info = self.extract_abstract_and_keywords(text_content)
        tags.extend(extracted_info['keywords'])
        
        return {
            "text": text_content,
            "embeddings": embeddings,
            "tags": list(set(tags)),
            "abstract": extracted_info['abstract']
        }
    
    def translate_text(self, text: str, target_lang='en') -> str:
        print(f"Übersetze Text in {target_lang}...")
        try:
            translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return translated_text
        except Exception as e:
            print(f"Fehler bei der Übersetzung: {e}")
            return text
    
    def get_ocr_text(self, image_path: str) -> str:
        """Führt OCR auf einem Bild durch."""
        try:
            return pytesseract.image_to_string(Image.open(image_path))
        except Exception as e:
            print(f"Fehler bei OCR: {e}")
            return ""