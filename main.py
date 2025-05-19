from llm_agentic_pdf_assistant.agents.loader_agent import load_and_split_pdf
from llm_agentic_pdf_assistant.agents.embedding_agent import EmbeddingAgent
from llm_agentic_pdf_assistant.agents.generator_agent import GeneratorAgent

def main():
    print("📄 Loading and chunking PDF...")
    chunks = load_and_split_pdf("pdf_data/EvolutionaryStrategies_Baeck.pdf")

    embedding_agent = EmbeddingAgent(
        model_name = "all-MiniLM-L6-v2",
        persist_directory = "./chroma_db"
    )
    generator_agent = GeneratorAgent(
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        local_path = "./local_model",
        save_model = True
    )
    print("🧠 Embedding and storing in vector DB...")
    embedding_agent.embed_chunks(chunks)
    print("🔄 Vector DB ready!")
    while True:
        query = input("\n💬 Ask a question (or type 'exit' to quit):\n> ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        print("🔍 Retrieving relevant context...")
        relevant_chunks = embedding_agent.query_index(query, k = 3)

        context = "\n".join(relevant_chunks)

        print("✍️ Generating answer...")
        answer = generator_agent.generate_response(context, query)
        print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    main()