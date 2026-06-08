import sys
import os
import json

sys.path.append('src')

from pdf_extractor import (
    extract_text_from_pdf,
    get_pdf_info
)
from classifier import classify_document
from entity_extractor import (
    extract_entities,
    save_entities
)
from rag_pipeline import setup_rag, answer_question

def process_document(pdf_path):
    """Complete pipeline"""
    
    print("="*50)
    print("DOCUMENT INTELLIGENCE SYSTEM")
    print("="*50)
    
    print("\n[1/4] Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    info = get_pdf_info(pdf_path)
    print(f"✓ {info['total_pages']} pages extracted")
    
    print("\n[2/4] Classifying...")
    classification = classify_document(text)
    doc_type = classification['document_type']
    print(f"✓ Type: {doc_type}")
    
    print("\n[3/4] Extracting entities...")
    entities = extract_entities(text, doc_type)
    print("✓ Entities extracted")
    
    print("\n[4/4] Setting up RAG...")
    index, chunks = setup_rag(text)
    print("✓ RAG ready")
    
    output_file = f"outputs/{info['filename']}_results.json"
    final_output = {
        "file_info": info,
        "classification": classification,
        "extracted_entities": entities
    }
    save_entities(final_output, output_file)
    
    print("\n" + "="*50)
    print(json.dumps(final_output, indent=4))
    
    return text, index, chunks, final_output


def chat_with_document(index, chunks):
    """Interactive chat"""
    print("\n" + "="*50)
    print("CHAT WITH YOUR DOCUMENT")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        if not question:
            continue
        
        answer = answer_question(
            question, index, chunks
        )
        print(f"\nAnswer: {answer}\n")
        print("-"*40)


if __name__ == "__main__":
    pdf_folder = "data/sample_pdfs"
    pdfs = [
        f for f in os.listdir(pdf_folder)
        if f.endswith('.pdf')
    ]
    
    if not pdfs:
        print("No PDFs in data/sample_pdfs/")
    else:
        pdf_path = os.path.join(
            pdf_folder, pdfs[0]
        )
        text, index, chunks, output = \
            process_document(pdf_path)
        chat_with_document(index, chunks)
