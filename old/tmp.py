#meant to keep this as main, but notebook was way easier for now

from src.vectorstore import VectorstoreHandler
from src.models import init_emb, init_llm


# Embedding models dictionary with unique identifiers as keys
# AVAILABLE_EMBS = {y
#     # OpenAI Embeddings
#     "openai-ada-002": lambda: OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")),
#     "openai-embedding-3-small": lambda: OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")),
#     "openai-embedding-3-large": lambda: OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY")),
    
#     # Hugging Face Embeddings
#     "hf-mpnet-base-v2": lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
#     "hf-minilm-l6-v2": lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
#     "hf-multiqa-minilm-l6-v1": lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-dot-v1")
# }

# # LLM mappings
# AVAILABLE_LLMS = {
#     "ChatGPT4o": "gpt-4o",
#     "ChatGPT3.5-turbo": "gpt-3.5-turbo",
#     "Llama3.2-3b": "llama3.2:3b",
# }

print("Hello world")


sources_dir = "sources"
subfolder_name = "book"
book_dir = sources_dir+"/"+subfolder_name


book_llm_name = "Llama3.2-3b"
book_emb_name = "hf-minilm-l6-v2"


book_emb = init_emb(book_emb_name)
book_llm = init_llm(book_llm_name)
handler = VectorstoreHandler(sources_dir, force_rebuild=False)
book_vs = handler.build_vectorstore(book_dir, book_emb)
book_retreiver = handler._init_retriever(book_vs, book_dir)