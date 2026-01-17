"""
Providers (LangChain based)

LLM:
- Groq -> llama-3.1-8b-instant

Embeddings:
- AWS Bedrock -> Amazon Titan Embeddings
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_aws import BedrockEmbeddings


# Load env vars from .env
load_dotenv()


# -------------------------
# LLM Provider (Groq)
# -------------------------
def get_llm_model() -> ChatGroq:
    """
    Returns a LangChain ChatGroq instance.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    llm_choice = os.getenv("LLM_CHOICE", "llama-3.1-8b-instant")

    return ChatGroq(
        groq_api_key=api_key,
        model=llm_choice,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    )


def get_ingestion_model() -> ChatGroq:
    """
    For ingestion tasks you can use the same model as LLM.
    """
    return get_llm_model()


# -------------------------
# Embedding Provider (Bedrock Titan)
# -------------------------
def get_embedding_client() -> BedrockEmbeddings:
    """
    Returns LangChain BedrockEmbeddings (Titan).
    This replaces openai.AsyncOpenAI from your older version.
    """
    region = os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("AWS_DEFAULT_REGION environment variable is required")

    # Titan embedding model id on AWS Bedrock
    embedding_model = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")

    return BedrockEmbeddings(
        model_id=embedding_model,
        region_name=region,
        # credentials automatically picked from env vars:
        # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    )


def get_embedding_model() -> str:
    """
    Returns the embedding model id (Titan).
    """
    return os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v1")


# -------------------------
# Validation + Info
# -------------------------
def validate_configuration() -> bool:
    """
    Validate required env variables.
    """
    required_vars = [
        "GROQ_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "DATABASE_URL",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        return False

    return True


def get_model_info() -> Dict[str, Any]:
    """
    Show which providers are active.
    """
    return {
        "llm_provider": "groq",
        "llm_model": os.getenv("LLM_CHOICE", "llama-3.1-8b-instant"),
        "embedding_provider": "aws_bedrock",
        "embedding_model": get_embedding_model(),
        "aws_region": os.getenv("AWS_DEFAULT_REGION"),
    }
