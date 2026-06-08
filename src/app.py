import streamlit as st
import sys
import os
import json

sys.path.append('src')

from pdf_extractor import (
    extract_text_from_pdf,
    get_pdf_info
)
from classifier import classify_document
from entity_extractor import extract_entities
from rag_pipeline import (
    build_multi_document_store,
    answer_from_multiple_docs
)
from insights_agent import run_insights_agent

st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Document Intelligence System v2.0")
st.markdown(
    "Upload multiple PDFs — AI will classify, "
    "extract entities, and answer questions "
    "across all documents"
)
st.divider()

if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'multi_index' not in st.session_state:
    st.session_state.multi_index = None
if 'multi_chunks' not in st.session_state:
    st.session_state.multi_chunks = None
if 'rag_ready' not in st.session_state:
    st.session_state.rag_ready = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files (upload multiple)",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(
            f"📁 {len(uploaded_files)} file(s) selected"
        )
        
        for f in uploaded_files:
            st.write(f"• {f.name}")
        
        if st.button(
            "🚀 Analyze All Documents",
            type="primary",
            use_container_width=True
        ):
            st.session_state.documents = []
            st.session_state.chat_history = []
            documents_for_rag = []
            
            for uploaded_file in uploaded_files:
                temp_path = f"data/sample_pdfs/{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):
                    text = extract_text_from_pdf(
                        temp_path
                    )
                    info = get_pdf_info(temp_path)
                    classification = classify_document(
                        text
                    )
                    doc_type = classification[
                        'document_type'
                    ]
                    entities = extract_entities(
                        text, doc_type
                    )
                
                with st.spinner(
                    f"🤖 Agent analyzing "
                    f"{uploaded_file.name}..."
                ):
                    insights = run_insights_agent(
                        doc_type, entities, text
                    )
                
                doc_info = {
                    "filename": uploaded_file.name,
                    "info": info,
                    "classification": classification,
                    "entities": entities,
                    "insights": insights,
                    "text": text,
                    "doc_type": doc_type
                }
                
                st.session_state.documents.append(
                    doc_info
                )
                documents_for_rag.append({
                    "filename": uploaded_file.name,
                    "doc_type": doc_type,
                    "text": text
                })
                
                st.success(
                    f"✅ {uploaded_file.name} → "
                    f"{doc_type.upper()}"
                )
            
            with st.spinner(
                "Building knowledge base..."
            ):
                index, chunks = \
                    build_multi_document_store(
                        documents_for_rag
                    )
                st.session_state.multi_index = index
                st.session_state.multi_chunks = chunks
                st.session_state.rag_ready = True
            
            st.success(
                f"✅ Knowledge base ready! "
                f"{len(uploaded_files)} docs indexed"
            )
    
    if st.session_state.documents:
        st.divider()
        st.subheader("📊 Document Analysis")
        
        for doc in st.session_state.documents:
            with st.expander(
                f"📄 {doc['filename']} — "
                f"{doc['doc_type'].upper()}",
                expanded=True
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Pages",
                        doc['info']['total_pages']
                    )
                with col2:
                    st.metric(
                        "Type",
                        doc['doc_type'].upper()
                    )
                with col3:
                    st.metric(
                        "Confidence",
                        doc['classification'][
                            'confidence'
                        ].upper()
                    )
                
                tab1, tab2, tab3 = st.tabs([
                    "🔍 Entities",
                    "🤖 AI Insights",
                    "⚠️ Risk Assessment"
                ])
                
                with tab1:
                    st.json(doc['entities'])
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json.dumps(
                            doc['entities'],
                            indent=4
                        ),
                        file_name=f"{doc['filename']}_entities.json",
                        mime="application/json",
                        key=f"dl_{doc['filename']}"
                    )
                
                with tab2:
                    if 'insights' in doc:
                        st.markdown(
                            "### 📋 Executive Summary"
                        )
                        st.write(
                            doc['insights'][
                                'executive_summary'
                            ]
                        )
                        st.divider()
                        st.markdown(
                            "### 💰 Financial Analysis"
                        )
                        st.write(
                            doc['insights'][
                                'financial_analysis'
                            ]
                        )
                        st.divider()
                        st.markdown(
                            "### ❓ Suggested Questions"
                        )
                        st.write(
                            doc['insights'][
                                'suggested_questions'
                            ]
                        )
                
                with tab3:
                    if 'insights' in doc:
                        st.markdown(
                            "### ⚠️ Risk Assessment"
                        )
                        st.write(
                            doc['insights'][
                                'risk_assessment'
                            ]
                        )

with right_col:
    st.subheader("💬 Chat Across All Documents")
    
    if not st.session_state.rag_ready:
        st.info(
            "👈 Upload and analyze PDFs first"
        )
        st.markdown("### Example questions:")
        st.markdown("""
        - *"What is the total invoice amount?"*
        - *"What are the candidate's key skills?"*
        - *"Who are the parties in the contract?"*
        - *"Compare dates across all documents"*
        """)
    else:
        st.success(
            f"✅ {len(st.session_state.documents)} "
            f"documents in knowledge base"
        )
        
        for doc in st.session_state.documents:
            st.write(
                f"📄 {doc['filename']} "
                f"({doc['doc_type']})"
            )
        
        st.divider()
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(
                    message['content']
                )
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    if 'sources' in message:
                        st.caption(
                            f"📚 Sources: "
                            f"{', '.join(message['sources'])}"
                        )
        
        question = st.chat_input(
            "Ask anything across all documents..."
        )
        
        if question:
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question
            })
            
            st.chat_message("user").write(question)
            
            with st.spinner("Searching..."):
                result = answer_from_multiple_docs(
                    question,
                    st.session_state.multi_index,
                    st.session_state.multi_chunks
                )
            
            with st.chat_message("assistant"):
                st.write(result['answer'])
                st.caption(
                    f"📚 Sources: "
                    f"{', '.join(result['sources'])}"
                )
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['answer'],
                'sources': result['sources']
            })

st.divider()
st.markdown(
    "🧠 Document Intelligence System v2.0 | "
    "Multi-Doc RAG | Agentic AI | Gemini | FAISS"
)
