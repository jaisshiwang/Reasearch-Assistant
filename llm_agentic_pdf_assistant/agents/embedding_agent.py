import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class EmbeddingAgent:
    """
    This class handles the embedding of chunks and querying the index.
    It uses ChromaDB to store and retrieve embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        """
        Initialize the embedding agent with a model name and persistence directory.
        Args:
            model_name (str): The name of the sentence transformer model to use for embeddings.
            persist_directory (str): The directory to store the ChromaDB database.
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        # Create a ChromaDB client and embedding function
        self.chroma_client = chromadb.PersistentClient(
            path = self.persist_directory,
        )
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name = self.model_name)

        # Create a collection for storing embeddings also known as a vector store or index
        # The collection is a key-value store where the key is the chunk ID and the value is the embedding
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_chunks",
            embedding_function=self.embedding_fn,
        )


    def embed_chunks(self, chunks):
        existing_ids = set(self.collection.get(include=[])["ids"])
        new_chunks = []

        for i, chunk in enumerate(chunks):
            doc_id = f"id_{i}"
            if doc_id not in existing_ids:
                new_chunks.append((doc_id, chunk))

        if new_chunks:
            ids, docs = zip(*new_chunks)
            self.collection.add(documents=list(docs), ids=list(ids))
            print(f"✅ Added {len(ids)} new chunks.")
        else:
            print("🟢 All chunks already embedded — skipping.")
            
            

    def query_index(self, query: str, k: int = 3):
        """
        Query the index for the most similar chunks to the query.
        Args:
            query (str): The query string.
            k (int): The number of similar chunks to return.
        Returns:
            list: List of similar chunks.
        """
        # No need to manually embed; Chroma does it using embedding_fn
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0]