from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def extract_entities(text, document_type):
    """
    Extract structured entities from document
    
    Args:
        text: extracted text from PDF
        document_type: type from classifier
    
    Returns:
        dictionary of extracted entities
    """
    
    if document_type == "invoice":
        fields = """
        - invoice_number
        - date
        - due_date
        - vendor_name
        - client_name
        - total_amount
        - line_items (list)
        """
    elif document_type == "resume":
        fields = """
        - full_name
        - email
        - phone
        - location
        - current_role
        - total_experience
        - skills (list)
        - education (list)
        - companies_worked (list)
        """
    elif document_type == "contract":
        fields = """
        - contract_title
        - parties_involved (list)
        - start_date
        - end_date
        - key_obligations (list)
        - payment_terms
        """
    elif document_type == "report":
        fields = """
        - report_title
        - author
        - date
        - key_findings (list)
        - recommendations (list)
        - summary
        """
    else:
        fields = """
        - title
        - date
        - author
        - key_points (list)
        - summary
        """
    
    prompt = f"""
    You are an expert document analyst.
    
    Extract the following fields from 
    this {document_type}:
    {fields}
    
    Document text:
    {text[:2000]}
    
    Rules:
    - Return ONLY valid JSON
    - If field not found write null
    - For lists return [] if not found
    - Extract exact values from document
    
    Return JSON only. No explanation.
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


def save_entities(entities, output_path):
    """
    Save extracted entities to JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(entities, f, indent=4)
    print(f"Entities saved to: {output_path}")


if __name__ == "__main__":
    test_resume = """
    Sai Sandeep Salina
    Data Scientist | AI Engineer
    Email: sai.sandeep@email.com
    Hyderabad, India
    Experience: 3.7 years
    Current: Accenture - Decision Science Analyst
    Previous: L&T - Design Engineer
    Skills: Python, ML, NLP, RAG, LangChain
    Education: B.Tech NIT Andhra Pradesh 8.35 CGPA
    """
    result = extract_entities(test_resume, "resume")
    print(json.dumps(result, indent=4))
