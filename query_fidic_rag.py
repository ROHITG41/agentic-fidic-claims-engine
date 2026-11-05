# =====================================================================
# SCRIPT: query_fidic_rag.py
# PURPOSE: End-to-End Enterprise RAG Adjudication Engine
# STACK: ChromaDB + gemini-embedding-2 + gemini-3.5-flash (Temp 0.0)
# =====================================================================

import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def run_fidic_rag_adjudication(claim_statement: str, persist_directory: str = "./chroma_fidic_db"):
    print("=" * 80)
    print(f"DISPUTE SUBMITTAL: \"{claim_statement}\"")
    print("=" * 80)

    # 1. RETRIEVE: Query ChromaDB using calibrated relevance threshold
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Calibrated gate: 0.45 threshold to capture strong legal matches from gemini-embedding-2
    results = vector_db.similarity_search_with_relevance_scores(claim_statement, k=3)
    
    retrieved_clauses = []
    print("\n[STEP 1: VECTOR RETRIEVAL AUDIT]")
    for doc, score in results:
        passed = score >= 0.45
        status = "PASSED GATE" if passed else "REJECTED (Score < 0.45)"
        clause_tag = doc.metadata.get("clause_number", doc.metadata.get("clause", "Clause"))
        title_tag = doc.metadata.get("clause_title", doc.metadata.get("title", "Contract Condition"))
        print(f"  • {clause_tag} ({title_tag}) | Score: {score:.4f} -> {status}")
        if passed:
            retrieved_clauses.append(doc)

    if not retrieved_clauses:
        print("\n[RESULT] No contract clauses satisfied the relevance threshold. Claim rejected.")
        return

    # 2. AUGMENT: Format retrieved chunks into an immutable ground-truth context block
    context_text = "\n\n".join([
        f"--- {doc.metadata.get('clause_number', 'Clause')}: {doc.metadata.get('clause_title', 'Title')} ---\n{doc.page_content}"
        for doc in retrieved_clauses
    ])

    # 3. GENERATE: Invoke Gemini 3.5 Flash with deterministic temperature=0.0
    print("\n[STEP 2: COGNITIVE ADJUDICATION (Gemini 3.5 Flash)]")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
    
    system_prompt = (
        "You are an impartial FIDIC Adjudicator operating under FIDIC 1999 Red Book rules.\n"
        "Strict Rule: You must ground your ruling EXCLUSIVELY on the provided Contract Excerpts.\n"
        "Do NOT invent clauses or cite rules outside this context.\n\n"
        f"VERIFIED CONTRACT EXCERPTS:\n{context_text}\n\n"
        f"CONTRACTOR CLAIM STATEMENT:\n{claim_statement}\n\n"
        "Provide your ruling in this structured format:\n"
        "1. GOVERNING SUB-CLAUSE(S): [State clause numbers retrieved]\n"
        "2. ENTITLEMENT VERDICT: [ENTITLED / TIME-BARRED / REJECTED]\n"
        "3. LEGAL REASONING: [Concise contractual rationale based strictly on the retrieved text]"
    )

    try:
        response = llm.invoke(system_prompt)
        print("\n" + "-" * 35 + " FORMAL ADJUDICATION RULING " + "-" * 35)
        print(response.content)
        print("-" * 98 + "\n")
    except Exception as exc:
        print(f"LLM API Error: {exc}")

if __name__ == "__main__":
    # Test Dispute 1: 35-day late notice
    run_fidic_rag_adjudication("Contractor submitted formal delay notice 35 days after flood event")
    
    # Test Dispute 2: Unforeseen physical obstruction
    run_fidic_rag_adjudication("Contractor encountered unexpectedly hard granite rock during station excavation")