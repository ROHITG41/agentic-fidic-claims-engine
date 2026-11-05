import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Page configuration
st.set_page_config(
    page_title="FIDIC Claims Engine | RAG Adjudication",
    page_icon="⚖️",
    layout="wide"
)

# Custom header styling
st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 700; color: #1B365D; margin-bottom: 0px; }
    .sub-title { font-size: 15px; color: #64748B; margin-bottom: 20px; }
    .verdict-box { background-color: #F8FAFC; border-left: 5px solid #1B365D; padding: 15px; border-radius: 4px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ FIDIC Enterprise Claims Engine (Red Book 1999)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Bridge 1: Persistent ChromaDB Vector Store • Calibrated Retrieval Gate • Grounded Deterministic Adjudication</div>', unsafe_allow_html=True)

# API Key Discovery: Supports Streamlit Secrets, environment variable, or manual input
api_key = os.environ.get("GEMINI_API_KEY")
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

with st.sidebar:
    st.header("Engine Architecture")
    st.markdown("**Vector Store:** ChromaDB (Persistent Disk)")
    st.markdown("**Embeddings:** `gemini-embedding-2`")
    st.markdown("**Cognitive Model:** `gemini-3.5-flash`")
    st.markdown("**Temperature:** `0.0` (Deterministic)")
    st.markdown("**Relevance Gate:** `0.45` (Cosine Similarity)")
    st.divider()
    
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
    else:
        st.success("API Key Authenticated")
        os.environ["GEMINI_API_KEY"] = api_key

    st.header("Pre-Engineered Scenarios")
    scenario = st.selectbox(
        "Load Test Submittal:",
        [
            "Custom Claim Submittal",
            "Dispute 1: Notice submitted 35 days after event (Time-Bar Check)",
            "Dispute 2: Unforeseen hard granite rock excavation (Sub-Clause 4.12)"
        ]
    )

# Input statement routing
default_text = ""
if scenario == "Dispute 1: Notice submitted 35 days after event (Time-Bar Check)":
    default_text = "Contractor submitted formal delay notice 35 days after flood event"
elif scenario == "Dispute 2: Unforeseen hard granite rock excavation (Sub-Clause 4.12)":
    default_text = "Contractor encountered unexpectedly hard granite rock during station excavation"

claim_statement = st.text_area("Contractor Claim Submittal Statement:", value=default_text, height=100)

if st.button("Adjudicate Claim Submittal", type="primary"):
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("API Key missing. Please set GEMINI_API_KEY in Streamlit Secrets or sidebar.")
        st.stop()

    if not claim_statement.strip():
        st.warning("Please enter a dispute statement.")
        st.stop()

    with st.spinner("Executing dense vector retrieval and cognitive adjudication..."):
        try:
            # 1. Connect to persistent ChromaDB
            persist_dir = "./chroma_fidic_db"
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
            vector_db = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )

            # 2. Vector search with calibrated gate (0.45)
            results = vector_db.similarity_search_with_relevance_scores(claim_statement, k=3)

            st.subheader("Step 1: Vector Retrieval & Relevance Audit")
            cols = st.columns(3)
            retrieved_clauses = []

            for idx, (doc, score) in enumerate(results):
                passed = score >= 0.45
                clause_tag = doc.metadata.get("clause_number", doc.metadata.get("clause", "Clause"))
                title_tag = doc.metadata.get("clause_title", doc.metadata.get("title", "Contract Condition"))

                if passed:
                    retrieved_clauses.append(doc)

                with cols[idx]:
                    st.metric(label=f"{clause_tag}", value=f"{score:.4f}", delta="Passed Gate" if passed else "Rejected", delta_color="normal" if passed else "inverse")
                    st.caption(f"**{title_tag}**")
                    st.text(doc.page_content[:140] + "...")

            if not retrieved_clauses:
                st.error("Relevance Gate Audit: No contract clauses exceeded the 0.45 threshold. Claim rejected without adjudication.")
                st.stop()

            # 3. Context Augmentation
            context_text = "\n\n".join([
                f"--- {doc.metadata.get('clause_number', 'Clause')}: {doc.metadata.get('clause_title', 'Title')} ---\n{doc.page_content}"
                for doc in retrieved_clauses
            ])

            # 4. Cognitive Adjudication (Gemini 3.5 Flash @ Temp 0.0)
            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
            
            system_prompt = (
                "You are an impartial FIDIC Adjudicator operating under FIDIC 1999 Red Book rules.\n"
                "Strict Rule: Ground your ruling EXCLUSIVELY on the provided Contract Excerpts.\n"
                "Do NOT invent clauses or cite rules outside this context.\n\n"
                f"VERIFIED CONTRACT EXCERPTS:\n{context_text}\n\n"
                f"CONTRACTOR CLAIM STATEMENT:\n{claim_statement}\n\n"
                "Provide your ruling in this structured format:\n"
                "1. GOVERNING SUB-CLAUSE(S): [State clause numbers retrieved]\n"
                "2. ENTITLEMENT VERDICT: [ENTITLED / TIME-BARRED / REJECTED]\n"
                "3. LEGAL REASONING: [Concise contractual rationale based strictly on the retrieved text]"
            )

            response = llm.invoke(system_prompt)
            
            # Clean extraction across SDK list/dict variations
            if isinstance(response.content, list):
                output_text = response.content[0].get("text", "")
            else:
                output_text = response.content

            st.subheader("Step 2: Formal Adjudication Ruling")
            st.markdown(f'<div class="verdict-box">{output_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Execution Error: {e}")