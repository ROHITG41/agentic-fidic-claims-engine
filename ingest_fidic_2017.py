import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

clauses_2017 = [
    Document(
        page_content=(
            "FIDIC 2017 Red Book Sub-Clause 20.2.1 (Notice of Claim): "
            "The claiming Party (whether Contractor or Employer) shall give a Notice to the Engineer, "
            "describing the event or circumstance giving rise to the cost, loss, delay or extension of Completion "
            "Period, no later than 28 days after the claiming Party became aware, or should have become aware, of the event. "
            "If the claiming Party fails to give a Notice of Claim within this period of 28 days, the Notice shall be invalid, "
            "the claiming Party shall not be entitled to any Extension of Time (EOT) or additional payment, and the other Party "
            "shall be discharged from any liability in connection with the event or circumstance."
        ),
        metadata={"edition": "2017", "clause_number": "Sub-Clause 20.2.1", "clause_title": "Notice of Claim (28-Day Reciprocal Time-Bar)"}
    ),
    Document(
        page_content=(
            "FIDIC 2017 Red Book Sub-Clause 20.2.4 (Fully Detailed Claim): "
            "The claiming Party shall submit to the Engineer a fully detailed Claim within 84 days after the claiming Party "
            "became aware, or should have become aware, of the event or circumstance. The fully detailed Claim shall include "
            "a detailed description of the event, contemporaneous records, and detailed contractual and legal basis for the "
            "EOT and/or additional payment claimed."
        ),
        metadata={"edition": "2017", "clause_number": "Sub-Clause 20.2.4", "clause_title": "Fully Detailed Claim (84 Days)"}
    ),
    Document(
        page_content=(
            "FIDIC 2017 Red Book Sub-Clause 3.7 (Agreement or Determination): "
            "When carrying out duties under Sub-Clause 3.7, the Engineer shall act neutrally between the Parties and shall "
            "not be deemed to act for the Employer. The Engineer shall give Notice of agreement or determination within 42 days. "
            "If the Engineer fails to issue a determination within 42 days, the claim is deemed rejected."
        ),
        metadata={"edition": "2017", "clause_number": "Sub-Clause 3.7", "clause_title": "Agreement or Determination (Neutrality & 42 Days)"}
    ),
    Document(
        page_content=(
            "FIDIC 2017 Red Book Sub-Clause 8.5 (Extension of Time for Completion): "
            "The Contractor shall be entitled subject to Sub-Clause 20.2 to an Extension of Time for Completion if and to the "
            "extent that completion is or will be delayed by any of the causes: Variations, exceptionally adverse climatic conditions, "
            "unforeseeable physical conditions under Sub-Clause 4.12, or delays caused by the Employer. Float buffers and concurrent "
            "contractor delays shall be audited under critical path delay principles."
        ),
        metadata={"edition": "2017", "clause_number": "Sub-Clause 8.5", "clause_title": "Extension of Time for Completion"}
    ),
    Document(
        page_content=(
            "FIDIC 2017 Red Book Sub-Clause 8.8 (Delay Damages): "
            "If the Contractor fails to comply with Sub-Clause 8.2 (Time for Completion), the Employer shall be entitled subject to "
            "Sub-Clause 20.2 to payment of Delay Damages by the Contractor. These delay damages shall be the daily sum stated in the "
            "Contract Data, subject to the maximum limit (cap) stated. The Employer is required to issue a Sub-Clause 20.2 Notice."
        ),
        metadata={"edition": "2017", "clause_number": "Sub-Clause 8.8", "clause_title": "Delay Damages"}
    ),
    Document(
        page_content=(
            "FIDIC 2017 Red Book Sub-Clause 21.4 (Obtaining DAAB's Decision): "
            "If a dispute arises between the Parties, either Party may refer the dispute in writing to the standing Dispute "
            "Avoidance/Adjudication Board (DAAB). The DAAB shall give its decision within 84 days after receiving the reference. "
            "The decision shall be binding on both Parties who shall promptly give effect to it."
        ),
        metadata={"edition": "2017", "clause_number": "Sub-Clause 21.4", "clause_title": "Obtaining DAAB's Decision (Standing DAAB)"}
    )
]

persist_directory = "./chroma_fidic_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

print(f"Ingesting {len(clauses_2017)} FIDIC 2017 clauses into {persist_directory}...")
vector_db.add_documents(clauses_2017)
print("Ingestion complete. ChromaDB now holds both FIDIC 1999 and FIDIC 2017 editions.")