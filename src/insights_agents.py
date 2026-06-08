from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)


def analyze_financial_data(entities_json):
    """Tool 1: Analyze financial information"""
    try:
        entities = json.loads(entities_json) \
            if isinstance(entities_json, str) \
            else entities_json

        prompt = f"""
        Analyze these financial entities and identify:
        1. Total amounts and breakdowns
        2. Payment dates and deadlines
        3. Any overdue or urgent items
        4. Cost saving opportunities
        
        Entities: {json.dumps(entities, indent=2)}
        
        Return clear financial summary with alerts.
        """
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Financial analysis: {str(e)}"


def identify_risks(entities_json):
    """Tool 2: Identify risks and flags"""
    try:
        entities = json.loads(entities_json) \
            if isinstance(entities_json, str) \
            else entities_json

        prompt = f"""
        Review these entities and identify:
        1. Urgent deadlines or due dates
        2. Missing or incomplete information
        3. Potential compliance issues
        4. Items requiring immediate attention
        5. Any anomalies or unusual values
        
        Entities: {json.dumps(entities, indent=2)}
        
        Format as risk report with severity:
        🔴 HIGH, 🟡 MEDIUM, 🟢 LOW
        """
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Risk analysis: {str(e)}"


def generate_summary(text_sample):
    """Tool 3: Generate executive summary"""
    try:
        prompt = f"""
        Generate concise executive summary 
        in 3-4 bullet points.
        
        Focus on:
        1. What this document is about
        2. Key parties involved
        3. Most important information
        4. Any action required
        
        Document:
        {text_sample[:1500]}
        
        Summary:
        """
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Summary: {str(e)}"


def suggest_questions(doc_type_and_entities):
    """Tool 4: Suggest follow-up questions"""
    try:
        prompt = f"""
        Based on this document suggest
        5 most relevant questions user 
        might want to ask.
        
        Make questions specific and useful.
        
        Document info: {doc_type_and_entities}
        
        Format as numbered list:
        1. Question one?
        2. Question two?
        """
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Questions: {str(e)}"


def run_insights_agent(doc_type, entities, text):
    """
    Main agent function
    Automatically analyzes document
    and generates comprehensive insights
    
    Args:
        doc_type: classified document type
        entities: extracted entities dict
        text: full document text
    
    Returns:
        dict with all insights
    """
    entities_str = json.dumps(entities, indent=2)
    doc_info = f"Type: {doc_type}\n{entities_str}"

    print(f"\nRunning insights agent "
          f"for {doc_type} document...")

    print("→ Generating summary...")
    summary = generate_summary(text)

    print("→ Analyzing financials...")
    financial = analyze_financial_data(
        entities_str
    )

    print("→ Identifying risks...")
    risks = identify_risks(entities_str)

    print("→ Generating questions...")
    questions = suggest_questions(doc_info)

    insights = {
        "document_type": doc_type,
        "executive_summary": summary,
        "financial_analysis": financial,
        "risk_assessment": risks,
        "suggested_questions": questions
    }

    print("✓ Insights agent complete")
    return insights
