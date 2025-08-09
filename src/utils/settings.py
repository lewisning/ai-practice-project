from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Application settings for the Tucows RAG Assistant.
    """
    app_name: str = "Tucows RAG Assistant"

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "kb_chunks"

    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY")  # Replace with your actual OpenAI API key
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
