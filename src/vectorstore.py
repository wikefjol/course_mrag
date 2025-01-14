import os
import json
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from src.utils import compute_folder_hash
from src.preprocessing import read_files
from tqdm import tqdm


class VectorstoreHandler:
    """
    A handler for managing multiple Chroma vectorstores across different subfolders.
    """
    def __init__(self, sources_directory, force_rebuild=False):
        """
        Initialize the VectorstoreHandler.

        Parameters:
            sources_directory (str): Directory containing subfolders for each vectorstore.
            force_rebuild (bool): Whether to rebuild vectorstores even if they exist.
        """
        self.sources_directory = sources_directory
        self.force_rebuild = force_rebuild
        self.vectorstores = {}  # To store vectorstore objects in memory

    def build_vectorstore(self, subfolder_path, embedding, embedding_name):
        """
        Build or load a vectorstore for a single subfolder.

        Parameters:
            subfolder_path (str): Path to the subfolder.
            embedding: The embedding model to use.

        Returns:
            Chroma: The initialized or loaded Chroma vectorstore.
        """
        # Compute folder hash to detect changes
        folder_hash = compute_folder_hash(subfolder_path)
        vectorstore_dir = os.path.join(subfolder_path, "chroma_store", embedding_name, folder_hash)

        # Metadata handling
        metadata_path = os.path.join(vectorstore_dir, "metadata.json")
        existing_metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)

        # Check if rebuild is needed
        expected_metadata = {"embedding_name": embedding_name, "folder_hash": folder_hash}
        rebuild_needed = (
            self.force_rebuild
            or not os.path.exists(vectorstore_dir)
            or existing_metadata != expected_metadata
        )

        if rebuild_needed:
            print(f"Rebuilding vectorstore for {subfolder_path}...")
            self._delete_vectorstore(vectorstore_dir)

            # Create a new vectorstore
            chunks = self._create_chunks(subfolder_path, embedding)
            print(f"Nr of chunks: {len(chunks)}")
            vec_store = Chroma(embedding_function=embedding, persist_directory=vectorstore_dir)

            print(f"Adding {len(chunks)} documents to the vectorstore...")
            vec_store.add_documents(chunks)

            # Save metadata
            os.makedirs(vectorstore_dir, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(expected_metadata, f, indent=4)

            return vec_store
        else:
            print(f"Loading existing vectorstore for {subfolder_path}...")
            return Chroma(embedding_function=embedding, persist_directory=vectorstore_dir)


    def _create_chunks(self, subfolder_path, embedding):
        """
        Split the text data in a subfolder into multiple semantic chunks for the vectorstore.

        Parameters:
            subfolder_path (str): Path to the subfolder.
            embedding: The embedding model to use.

        Returns:
            list: List of text chunks.
        """
        chunks = []
        sem_text_splitter = SemanticChunker(embedding)

        print(f"Splitting text in {subfolder_path} into chunks...")
        dataset = read_files(subfolder_path)

        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing documents"):
            text_chunks = sem_text_splitter.create_documents([row["text"]])
            chunks.extend(text_chunks)
        return chunks

    def _delete_vectorstore(self, vectorstore_dir): 
        """
        Delete a vectorstore directory and its contents.

        Parameters:
            vectorstore_dir (str): Path to the vectorstore directory.
        """
        if os.path.exists(vectorstore_dir):
            for root, dirs, files in os.walk(vectorstore_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(vectorstore_dir)

    def _init_retriever(self, vectorstore, subfolder_name, k):
        """
        Initialize retrievers for all vectorstores.

        Parameters:
            k (int): Number of documents to retrieve.

        Returns:
            dict: Dictionary of subfolder names to retriever objects.
        """
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def init_retrievers(self, vectorstore, k):
        """
        Initialize retrievers for all vectorstores.

        Parameters:
            k (int): Number of documents to retrieve.

        Returns:
            dict: Dictionary of subfolder names to retriever objects.
        """
        retrievers = {}
        for subfolder_name, vectorstore in self.vectorstores.items():
            retrievers[subfolder_name] = vectorstore.as_retriever(search_kwargs={"k": k})

        return retrievers
