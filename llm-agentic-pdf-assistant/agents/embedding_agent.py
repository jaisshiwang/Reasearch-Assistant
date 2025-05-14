import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Create a ChromaDB client and embedding function
chroma_client = chromadb.Client()
model_name = "all-MiniLM-L6-v2"
embedding_fn = SentenceTransformerEmbeddingFunction(model_name = model_name)

# Create a collection for storing embeddings also known as a vector store or index
# The collection is a key-value store where the key is the chunk ID and the value is the embedding
collection = chroma_client.create_collection(
    name="pdf_chunks",
    embedding_function=embedding_fn,
)


def embed_chunks(chunks):
    """
    Embed the chunks and add them to the ChromaDB collection.
    Args:
        chunks (list): List of text chunks to embed.
    """
    for i, chunk in enumerate(chunks):
        # Embed the chunk and add it to the collection
        collection.add(
            documents=[chunk],
            ids=[f"id_{i}"]
        )

def query_index(query : str, k : int = 3):
    """
    Query the index for the most similar chunks to the query.
    Args:
        query (str): The query string.
        k (int): The number of similar chunks to return.
    Returns:
        list: List of similar chunks.
    """
    results = collection.query(
        query_embeddings=[query],
        n_results=k
    )
    return results