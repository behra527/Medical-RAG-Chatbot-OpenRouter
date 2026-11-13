"""Explicit configuration; API key is never logged or returned to clients."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv('OPENROUTER_API_KEY', ''))
    chat_model: str = field(default_factory=lambda: os.getenv('OPENROUTER_CHAT_MODEL', 'openai/gpt-4o-mini'))
    embed_model: str = field(default_factory=lambda: os.getenv('OPENROUTER_EMBED_MODEL', 'openai/text-embedding-3-small'))
    base_url: str = field(default_factory=lambda: os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/'))
    db_path: str = field(default_factory=lambda: os.getenv('RAG_DB_PATH', 'data/medical_rag.sqlite3'))
    pdf_dir: str = field(default_factory=lambda: os.getenv('RAG_PDF_DIR', 'data/guidelines'))
    min_similarity: float = field(default_factory=lambda: float(os.getenv('MIN_SIMILARITY', '.40')))
    chunk_words: int = field(default_factory=lambda: int(os.getenv('CHUNK_WORDS', '190')))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv('CHUNK_OVERLAP', '40')))
    top_k: int = field(default_factory=lambda: int(os.getenv('TOP_K', '5')))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv('MAX_UPLOAD_MB', '20')))
    admin_api_key: str = field(default_factory=lambda: os.getenv('ADMIN_API_KEY', ''))

    def validate(self):
        if not 0 <= self.min_similarity <= 1:
            raise ValueError('MIN_SIMILARITY must be within [0,1]')
        if self.chunk_words < 40 or not 0 <= self.chunk_overlap < self.chunk_words:
            raise ValueError('CHUNK_WORDS must be >=40 and 0 <= CHUNK_OVERLAP < CHUNK_WORDS')
        if not 1 <= self.top_k <= 20:
            raise ValueError('TOP_K must be between 1 and 20')
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.pdf_dir).mkdir(parents=True, exist_ok=True)
        return self
