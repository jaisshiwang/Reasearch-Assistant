from llm_agentic_pdf_assistant.agents.loader_agent import load_and_split_pdf
from llm_agentic_pdf_assistant.agents.embedding_agent import embed_chunks, query_index
from llm_agentic_pdf_assistant.agents.generator_agent import generate_response

def main():
    print("📄 Loading and chunking PDF...")
    chunks = load_and_split_pdf("pdf_data/EvolutionaryStrategies_Baeck.pdf")

    print("🧠 Embedding and storing in vector DB...")
    embed_chunks(chunks)
    print("🔄 Vector DB ready!")
    while True:
        query = input("\n💬 Ask a question (or type 'exit' to quit):\n> ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        print("🔍 Retrieving relevant context...")
        relevant_chunks = query_index(query, k = 3)

        context = "\n".join(relevant_chunks)

        print("✍️ Generating answer...")
        answer = generate_response(context, query)
        print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    main()