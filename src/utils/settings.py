from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings for the Tucows RAG Assistant.
    """
    app_name: str = "Tucows RAG Assistant"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "kb_chunks"
    embedding_dim: int = 1536
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
