import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF file
    
    Args:
        pdf_path: path to the PDF file
    
    Returns:
        extracted text as string
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at: {pdf_path}"
        )
    
    document = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(document)):
        page = document[page_num]
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n"
        full_text += text
    
    document.close()
    return full_text.strip()


def get_pdf_info(pdf_path):
    """
    Get basic info about PDF
    """
    document = fitz.open(pdf_path)
    
    info = {
        "filename": os.path.basename(pdf_path),
        "total_pages": len(document),
        "file_size_kb": round(
            os.path.getsize(pdf_path) / 1024, 2
        )
    }
    
    document.close()
    return info


if __name__ == "__main__":
    print("PDF Extractor ready")
