"""
=====================================================================
Loom AI - Document Retriever Service | Configuration Module
=====================================================================
Loads environment variables from .env.live or .env.dev depending on
the ENVIRONMENT system variable. Uses pydantic-settings for type-safe
config with automatic env-var binding.

Environment selection logic:
  1. Check the ENVIRONMENT env var (defaults to "live")
  2. Load the matching .env.<environment> file
  3. All settings can be overridden via environment variables
=====================================================================
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Determine the project root directory (where .env files live).
# This ensures .env files are found regardless of the working directory
# (e.g., when running from an IDE or a different folder).
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Determine which .env file to load based on the ENVIRONMENT variable.
# If ENVIRONMENT is not set, default to "live" (production).
# This lets us switch between .env.live and .env.dev seamlessly.
# ---------------------------------------------------------------------------
_environment = os.getenv("ENVIRONMENT", "live")
_env_file = _project_root / f".env.{_environment}"
_env_fallback = _project_root / ".env"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Each field maps to an env var with the same name (case-insensitive).
    Example: doc_retriever_port -> DOC_RETRIEVER_PORT
    """

    # --- General -----------------------------------------------------------
    # "live" or "dev" - controls reload behavior and logging level
    environment: str = "live"

    # Port on which the FastAPI server will listen
    doc_retriever_port: int = 8001

    # --- Weaviate (Vector Database) ----------------------------------------
    # URL for the Weaviate REST API (self-hosted)
    weaviate_url: str = "http://localhost:8080"

    # gRPC port for Weaviate (used for faster batch operations)
    weaviate_grpc_port: int = 50051

    # --- OpenAI (Embeddings) -----------------------------------------------
    # API key for OpenAI - used to generate text embeddings
    openai_api_key: str = ""

    # The embedding model to use (text-embedding-3-small is cost-effective)
    embedding_model: str = "text-embedding-3-small"

    # --- Chunking ----------------------------------------------------------
    # Maximum number of tokens per text chunk (for splitting large documents)
    chunk_size: int = 500

    # Number of overlapping tokens between consecutive chunks
    # This ensures context isn't lost at chunk boundaries
    chunk_overlap: int = 50

    # --- pydantic-settings config ------------------------------------------
    # Load environment-specific file first (.env.live or .env.dev),
    # then fall back to .env for any values not set (like the API key).
    # Later files in the tuple have lower priority (used as fallback).
    model_config = {
        "env_file": (_env_file, _env_fallback),
        "env_file_encoding": "utf-8",
    }


# ---------------------------------------------------------------------------
# Singleton settings instance - import this throughout the application.
# Usage: from app.config import settings
# ---------------------------------------------------------------------------
settings = Settings()
