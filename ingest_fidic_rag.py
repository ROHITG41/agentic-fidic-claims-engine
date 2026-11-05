# =====================================================================
# SCRIPT: ingest_fidic_rag.py
# PURPOSE: Semantic Chunking & Vector Ingestion of FIDIC Contract Clauses
# STACK: Python 3.12 / LangChain / ChromaDB / Google text-embedding-004
# =====================================================================

import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. SAMPLE FIDIC CLAUSE CORPUS (Red Book 1999 Baseline)
FIDIC_CONTRACT_CORPUS = [
    {
        "clause": "Sub-Clause 20.1",
        "title": "Contractor's Claims",
        "text": (
            "If the Contractor considers himself to be entitled to any extension of the Time for Completion "
            "and/or any additional payment, under any Clause of these Conditions or otherwise in connection with "
            "the Contract, the Contractor shall give notice to the Engineer, describing the event or circumstance "
            "giving rise to the claim. The notice shall be given as soon as practicable, and not later than 28 days "
            "after the Contractor became aware, or should have become aware, of the event or circumstance. "
            "If the Contractor fails to give notice of a claim within such period of 28 days, the Time for Completion "
            "shall not be extended, the Contractor shall not be entitled to additional payment, and the Employer "
            "shall be discharged from all liability in connection with the claim."
        )
    },
    {
        "clause": "Sub-Clause 8.4",
        "title": "Extension of Time for Completion",
        "text": (
            "The Contractor shall be entitled subject to Sub-Clause 20.1 to an extension of the Time for Completion "
            "if and to the extent that completion for the purposes of Sub-Clause 10.1 is or will be delayed by any of "
            "the following causes: (a) a Variation, (b) an exceptionally adverse climatic condition, (c) unforeseeable "
            "shortages in the availability of personnel or Goods caused by epidemic or governmental actions, or "
            "(d) any delay, impediment or prevention caused by or attributable to the Employer, the Employer's Personnel, "
            "or the Employer's other contractors on the Site."
        )
    },
    {
        "clause": "Sub-Clause 4.12",
        "title": "Unforeseeable Physical Conditions",
        "text": (
            "If the Contractor encounters adverse physical conditions which he considers to have been Unforeseeable, "
            "the Contractor shall give notice to the Engineer as soon as practicable. This notice shall describe the "
            "physical conditions, so that they can be inspected by the Engineer, and shall set out the reasons why the "
            "Contractor considers them to be Unforeseeable. If the Contractor suffers delay and/or incurs Cost due to "
            "these conditions, he shall be entitled subject to Sub-Clause 20.1 to: (a) an extension of time for any such "
            "delay, if completion is or will be delayed, and (b) payment of any such Cost, which shall be included in "
            "the Contract Price."
        )
    },
    {
        "clause": "Sub-Clause 3.5",
        "title": "Determinations",
        "text": (
            "Whenever these Conditions provide that the Engineer shall proceed in accordance with this Sub-Clause 3.5 "
            "to agree or determine any matter, the Engineer shall consult with each Party in an endeavour to reach agreement. "
            "If agreement is not achieved, the Engineer shall make a fair determination in accordance with the Contract, "
            "taking due regard of all relevant circumstances. The Engineer shall give notice to both Parties of each "
            "agreement or determination, with supporting particulars, within 28 days from the receipt of the corresponding "
            "claim or request."
        )
    }
]

def build_fidic_vector_database(persist_directory: str = "./chroma_fidic_db") -> Chroma:
    """
    Parses contract clauses, chunks them with boundary respect,
    generates 768-dimensional embeddings, and writes to local disk.
    """
    print("[1/3] Preparing documents and metadata...")
    documents: List[Document] = []
    
    # Text splitter designed to prevent cutting mid-sentence
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    for item in FIDIC_CONTRACT_CORPUS:
        # Split text while inheriting the clause reference
        chunks = splitter.split_text(item["text"])
        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "clause_number": item["clause"],
                    "clause_title": item["title"],
                    "chunk_id": f"{item['clause']}_part_{idx + 1}"
                }
            )
            documents.append(doc)
            
    print(f"      Total semantic chunks created: {len(documents)}")

    print("[2/3] Generating dense embeddings via Google text-embedding-004...")
    # Requires GEMINI_API_KEY in environment variables
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

    print(f"[3/3] Writing vectors to persistent storage: {persist_directory}...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("SUCCESS: Vector database compiled and saved to disk.")
    return vector_db

if __name__ == "__main__":
    build_fidic_vector_database()