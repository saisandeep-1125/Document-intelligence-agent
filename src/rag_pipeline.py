from google import genai
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Loading embedding model...")
embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)
print("Embedding model ready.")


def chunk_text(text, chunk_size=500, overlap=50):
    """Split document into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_multi_document_store(documents):
    """
    Build vector store for multiple documents
    
    Args:
        documents: list of dicts with
        filename, doc_type, text
    
    Returns:
        index, all_chunks with metadata
    """
    print(
        f"Building multi-document store "
        f"for {len(documents)} documents..."
    )
    
    all_chunks = []
    all_embeddings = []
    
    for doc in documents:
        chunks = chunk_text(doc['text'])
        
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "filename": doc['filename'],
                "doc_type": doc['doc_type']
            })
        
        texts = [c for c in chunks]
        embeddings = embedding_model.encode(texts)
        all_embeddings.append(embeddings)
        
        print(
            f"✓ Processed {doc['filename']} "
            f"→ {len(chunks)} chunks"
        )
    
    all_embeddings = np.vstack(all_embeddings)
    all_embeddings = all_embeddings.astype('float32')
    
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)
    
    print(f"✓ Multi-document store ready")
    print(f"  Total chunks: {index.ntotal}")
    
    return index, all_chunks


def retrieve_from_multiple_docs(
    query, index, all_chunks, k=4
):
    """
    Retrieve relevant chunks across documents
    """
    query_embedding = embedding_model.encode(
        [query]
    )
    query_embedding = np.array(
        query_embedding, dtype='float32'
    )
    
    distances, indices = index.search(
        query_embedding, k
    )
    
    relevant = []
    for i in indices[0]:
        if i < len(all_chunks):
            relevant.append(all_chunks[i])
    
    return relevant


def answer_from_multiple_docs(
    query, index, all_chunks
):
    """
    RAG answer with source attribution
    """
    relevant_chunks = retrieve_from_multiple_docs(
        query, index, all_chunks
    )
    
    context_parts = []
    sources_used = set()
    
    for chunk in relevant_chunks:
        context_parts.append(
            f"[From {chunk['filename']}]:\n"
            f"{chunk['text']}"
        )
        sources_used.add(chunk['filename'])
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""
    You are a helpful document assistant.
    Answer based ONLY on context below.
    
    If answer not found say:
    "I could not find this information 
    in any of the uploaded documents."
    
    Always mention which document 
    your answer comes from.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer (mention source document):
    """
    
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )
    
    return {
        "answer": response.text.strip(),
        "sources": list(sources_used)
    }


def setup_rag(text):
    """Single document RAG setup"""
    chunks_text = chunk_text(text)
    embeddings = embedding_model.encode(chunks_text)
    embeddings = np.array(
        embeddings, dtype='float32'
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks_text


def answer_question(query, index, chunks):
    """Single document answer"""
    query_embedding = embedding_model.encode(
        [query]
    )
    query_embedding = np.array(
        query_embedding, dtype='float32'
    )
    distances, indices = index.search(
        query_embedding, 3
    )
    relevant_chunks = [
        chunks[i] for i in indices[0]
    ]
    context = "\n\n".join(relevant_chunks)
    prompt = f"""
    Answer based ONLY on context below.
    If not found say "Not found in document."
    
    Context: {context}
    Question: {query}
    Answer:
    """
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )
    return response.text.strip()
