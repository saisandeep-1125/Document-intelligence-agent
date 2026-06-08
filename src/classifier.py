from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def classify_document(text):
    """
    Classify document into type using Gemini
    
    Args:
        text: extracted text from PDF
    
    Returns:
        dictionary with document_type 
        and confidence
    """
    sample_text = text[:1000]
    
    prompt = f"""
    You are a document classification expert.
    
    Classify the following document into 
    ONE of these categories:
    - invoice
    - contract
    - report
    - email
    - resume
    - other
    
    Document text:
    {sample_text}
    
    Respond in JSON format only. No other text.
    Example:
    {{"document_type": "invoice", 
      "confidence": "high", 
      "reason": "contains payment terms"}}
    """
    
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )
    
    response_text = response.text.strip()
    response_text = response_text.replace(
        "```json", ""
    )
    response_text = response_text.replace(
        "```", ""
    )
    response_text = response_text.strip()
    
    result = json.loads(response_text)
    return result


if __name__ == "__main__":
    test_text = """
    INVOICE #12345
    Date: January 15, 2024
    Bill To: ABC Company
    Amount Due: $5,420.00
    Payment Due: February 15, 2024
    """
    result = classify_document(test_text)
    print(f"Type: {result['document_type']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason: {result['reason']}")
