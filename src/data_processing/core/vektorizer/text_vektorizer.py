from sentence_transformers import SentenceTransformer

class TextVectorizer:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> list:
        return self.model.encode(text).tolist()