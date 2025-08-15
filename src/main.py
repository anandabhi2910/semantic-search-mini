import os
import sys
from sentence_transformers import SentenceTransformer
from text_processor import read_and_preprocess_documents
from embedding_generator import generate_and_save_embeddings
from similarity_search import load_data, search_documents

# --- Configuration ---
# Get the path to the 'data/raw' directory relative to the current script
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'embeddings.npy')

def setup_and_run_pipeline():
    """Checks for existing embeddings and runs the full pipeline if needed."""
    if not os.path.exists(EMBEDDINGS_PATH):
        print("Embeddings not found. Running full pipeline...")
        # 1. Read and preprocess documents
        documents = read_and_preprocess_documents(RAW_DATA_PATH)
        if not documents:
            sys.exit("Failed to process documents. Exiting.")
        # 2. Generate and save embeddings
        generate_and_save_embeddings(documents)
    
    print("Setup complete. Ready for search!")
    
def main():
    """Main function to run the semantic search application."""
    
    # Run the setup pipeline
    setup_and_run_pipeline()
    
    # Load the pre-trained model and data
    print("\nLoading Sentence-Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, documents = load_data()
    
    if embeddings is None or documents is None:
        sys.exit("Failed to load data. Exiting.")
    
    print("\n--- Semantic Search Engine Ready ---")
    print("Type your query and press Enter. Type 'exit' to quit.")
    
    while True:
        query = input("Your query: ")
        if query.lower() == 'exit':
            print("Exiting application. Goodbye!")
            break
            
        if not query.strip():
            print("Please enter a valid query.")
            continue
            
        results = search_documents(query, model, embeddings, documents)
        
        print("\n--- Top Results ---")
        for i, (doc, score) in enumerate(results):
            print(f"Rank {i+1} (Score: {score:.4f}):")
            print(f"{doc[:200]}...") # Print a snippet
        print("-" * 20)

if __name__ == "__main__":
    main()