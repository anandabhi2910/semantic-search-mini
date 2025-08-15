import os
import numpy as np
from sentence_transformers import SentenceTransformer
from text_processor import read_and_preprocess_documents # Import for initial demo purposes

# --- Configuration ---
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'embeddings.npy')
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'documents.txt')

def load_data():
    """Loads the pre-computed embeddings and documents."""
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(DOCUMENTS_PATH, 'r', encoding='latin-1') as f:
            documents = f.read().splitlines()
        print("Embeddings and documents loaded successfully.")
        return embeddings, documents
    except FileNotFoundError:
        print("Error: Embeddings or documents file not found. Please run embedding_generator.py first.")
        return None, None

def search_documents(query, model, embeddings, documents, top_k=3):
    """
    Performs a semantic search for a given query.

    Args:
        query (str): The search query.
        model (SentenceTransformer): The model to encode the query.
        embeddings (np.array): The embeddings of all documents.
        documents (list): The original text documents.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of (document, score) tuples.
    """
    # Encode the query
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    # Calculate cosine similarity between the query and all documents
    # Using dot product for normalized vectors is equivalent to cosine similarity
    similarities = np.dot(query_embedding, embeddings.T)
    
    # Get the indices of the top-k most similar documents
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Get the top-k documents and their scores
    results = [(documents[i], similarities[i]) for i in top_k_indices]
    
    return results

if __name__ == "__main__":
    # Load the pre-trained model
    print("Loading Sentence-Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load embeddings and documents
    document_embeddings, document_texts = load_data()
    
    if document_embeddings is not None and document_texts is not None:
        # Example search
        search_query = "What is Tokenization?"
        print(f"\nSearching for: '{search_query}'")
        
        results = search_documents(search_query, model, document_embeddings, document_texts)
        
        print("\n--- Top 3 Results ---")
        for i, (doc, score) in enumerate(results):
            print(f"Rank {i+1} (Score: {score:.4f}):")
            print(f"{doc[:500]}...\n") # Print a snippet of the document
