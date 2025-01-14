import os
from typing import Dict, Any
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Embedding models dictionary with unique identifiers as keys
AVAILABLE_EMBS = {
    # OpenAI Embeddings
    "openai-ada-002": lambda: OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")),
    "openai-embedding-3-small": lambda: OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")),
    "openai-embedding-3-large": lambda: OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY")),
    
    # Hugging Face Embeddings
    "hf-mpnet-base-v2": lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    "hf-minilm-l6-v2": lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "hf-multiqa-minilm-l6-v1": lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-dot-v1")
}

# LLM mappings
AVAILABLE_LLMS = {
    "ChatGPT4o": "gpt-4o",
    "ChatGPT3.5-turbo": "gpt-3.5-turbo",
    "Llama3.2-3b": "llama3.2:3b",
}

def init_llm(model_name: str):
    """
    Initialize a large language model (LLM).

    Args:
        model_name: Name of the LLM.

    Returns:
        Configured LLM instance.

    Raises:
        ValueError: If the LLM model is not supported.
    """
    if model_name.startswith("ChatGPT"):
        from langchain_openai.chat_models import ChatOpenAI
        return ChatOpenAI(model=AVAILABLE_LLMS[model_name])
    elif model_name.startswith("Llama"):
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=AVAILABLE_LLMS[model_name])
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

def init_emb(identifier: str) -> Any:
    """
    Initialize an embedding model based on a unique identifier.

    Args:
        identifier: Unique identifier for the embedding model (e.g., 'openai-ada-002').

    Returns:
        Configured embedding model.

    Raises:
        ValueError: If the identifier is not found in AVAILABLE_EMBS.
    """
    if identifier not in AVAILABLE_EMBS:
        raise ValueError(f"Unsupported embedding model identifier: {identifier}")
    try:
        return AVAILABLE_EMBS[identifier]()
    except Exception as e:
        raise RuntimeError(f"Error initializing embedding model '{identifier}': {e}")
