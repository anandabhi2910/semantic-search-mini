import os
import numpy as np
from sentence_transformers import SentenceTransformer
from text_processor import read_and_preprocess_documents

# --- Configuration ---
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')# Adjusted for notebook compatibility
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'embeddings.npy')
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'documents.txt')

def generate_and_save_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for a list of documents and saves them to a file.

    Args:
        documents (list): A list of cleaned document strings.
        model_name (str): The name of the SentenceTransformer model to use.
    """
    print(f"Loading the SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, convert_to_numpy=True)

    print(f"Embeddings generated with shape: {embeddings.shape}")

    #Ensure the processed directory exists
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    
    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)
    
    print("Embeddings saved successfully.")
    return embeddings

if __name__ == "__main__":
    # Read the cleaned documents
    documents = read_and_preprocess_documents(RAW_DATA_PATH)

    if not documents:
        print("No documents to process. Exiting.")
    else:
        # Save the cleaned documents to a text file
        # os.makedirs(os.path.dirname(DOCUMENTS_PATH), exist_ok=True)
        embeddings = generate_and_save_embeddings(documents)
        with open(DOCUMENTS_PATH, 'w', encoding='latin-1') as f:
            for doc in documents:
                f.write(doc + "\n")

        print(f"Original documents saved to {DOCUMENTS_PATH}.")
