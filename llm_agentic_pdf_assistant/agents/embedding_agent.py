import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Create a ChromaDB client and embedding function
chroma_client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="./chroma_db",  # Directory to store the database
    )
)
model_name = "all-MiniLM-L6-v2"
embedding_fn = SentenceTransformerEmbeddingFunction(model_name = model_name)

# Create a collection for storing embeddings also known as a vector store or index
# The collection is a key-value store where the key is the chunk ID and the value is the embedding
collection = chroma_client.get_or_create_collection(
    name="pdf_chunks",
    embedding_function=embedding_fn,
)


def embed_chunks(chunks):
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = []

    for i, chunk in enumerate(chunks):
        doc_id = f"id_{i}"
        if doc_id not in existing_ids:
            new_chunks.append((doc_id, chunk))

    if new_chunks:
        ids, docs = zip(*new_chunks)
        collection.add(documents=list(docs), ids=list(ids))
        print(f"✅ Added {len(ids)} new chunks.")
    else:
        print("🟢 All chunks already embedded — skipping.")
        
        

def query_index(query: str, k: int = 3):
    """
    Query the index for the most similar chunks to the query.
    Args:
        query (str): The query string.
        k (int): The number of similar chunks to return.
    Returns:
        list: List of similar chunks.
    """
    # No need to manually embed; Chroma does it using embedding_fn
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results['documents'][0]