import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = input("Enter your Gemini API Key: ").strip()
    os.environ["GEMINI_API_KEY"] = api_key

persist_directory = "./chroma_fidic_db"

# Clear out any corrupted or partial database files
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

all_clauses = [
    # --- FIDIC 1999 RED BOOK CLAUSES ---
    Document(
        page_content="FIDIC 1999 Red Book Sub-Clause 20.1 (Contractor's Claims): If the Contractor considers himself entitled to any extension of the Time for Completion and/or any additional payment, the Contractor shall give notice to the Engineer, describing the event or circumstance giving rise to the claim. The notice shall be given as soon as practicable, and not later than 28 days after the Contractor became aware, or should have become aware, of the event or circumstance. If the Contractor fails to give notice of a claim within such period of 28 days, the Time for Completion shall not be extended, the Contractor shall not be entitled to additional payment, and the Employer shall be discharged from all liability in connection with the claim.",
        metadata={"edition": "1999", "clause_number": "Sub-Clause 20.1", "clause_title": "Contractor's Claims (28-Day Condition Precedent)"}
    ),
    Document(
        page_content="FIDIC 1999 Red Book Sub-Clause 4.12 (Unforeseeable Physical Conditions): In this Sub-Clause, 'physical conditions' means natural physical conditions and man-made and other physical obstructions and pollutants, which the Contractor encounters at the Site when executing the Works, including sub-surface and hydrological conditions but excluding climatic conditions. If the Contractor encounters adverse physical conditions which he considers to have been Unforeseeable, the Contractor shall give notice to the Engineer as soon as practicable.",
        metadata={"edition": "1999", "clause_number": "Sub-Clause 4.12", "clause_title": "Unforeseeable Physical Conditions"}
    ),
    Document(
        page_content="FIDIC 1999 Red Book Sub-Clause 8.4 (Extension of Time for Completion): The Contractor shall be entitled subject to Sub-Clause 20.1 to an extension of the Time for Completion if and to the extent that completion is or will be delayed by any of the following causes: a Variation, exceptionally adverse climatic conditions, Unforeseeable physical conditions under Sub-Clause 4.12, or any delay, impediment or prevention caused by the Employer. Available total float on the critical path and concurrent contractor delay shall be deducted under SCL Delay Protocol principles.",
        metadata={"edition": "1999", "clause_number": "Sub-Clause 8.4", "clause_title": "Extension of Time for Completion"}
    ),
    Document(
        page_content="FIDIC 1999 Red Book Sub-Clause 8.7 (Delay Damages): If the Contractor fails to comply with Sub-Clause 8.2 (Time for Completion), the Contractor shall pay delay damages to the Employer for this default. These delay damages shall be the sum stated in the Appendix to Tender, which shall be paid for every day between the relevant Time for Completion and the date stated in the Taking-Over Certificate. However, the total amount due under this Sub-Clause shall not exceed the maximum amount of delay damages stated in the Appendix to Tender.",
        metadata={"edition": "1999", "clause_number": "Sub-Clause 8.7", "clause_title": "Delay Damages (Liquidated Damages)"}
    ),
    Document(
        page_content="FIDIC 1999 Red Book Sub-Clause 3.5 (Determinations): Whenever these Conditions provide that the Engineer shall proceed in accordance with this Sub-Clause 3.5 to agree or determine any matter, the Engineer shall consult with each Party in an endeavour to reach agreement. If agreement is not reached, the Engineer shall make a fair determination in accordance with the Contract, taking due regard of all relevant circumstances.",
        metadata={"edition": "1999", "clause_number": "Sub-Clause 3.5", "clause_title": "Engineer's Determinations"}
    ),
    Document(
        page_content="FIDIC 1999 Red Book Sub-Clause 20.4 (Obtaining Dispute Adjudication Board's Decision): If a dispute arises between the Parties in connection with, or arising out of, the Contract or the execution of the Works, either Party may refer the dispute in writing to the DAB for its decision. The DAB shall give its decision within 84 days after receiving such reference.",
        metadata={"edition": "1999", "clause_number": "Sub-Clause 20.4", "clause_title": "Obtaining Dispute Adjudication Board's Decision"}
    ),

    # --- FIDIC 2017 SECOND EDITION CLAUSES ---
    Document(
        page_content="FIDIC 2017 Red Book Sub-Clause 20.2.1 (Notice of Claim): The claiming Party (whether Contractor or Employer) shall give a Notice to the Engineer, describing the event or circumstance giving rise to the cost, loss, delay or extension of Completion Period, no later than 28 days after the claiming Party became aware, or should have become aware, of the event. If the claiming Party fails to give a Notice of Claim within this period of 28 days, the Notice shall be invalid, the claiming Party shall not be entitled to any Extension of Time (EOT) or additional payment, and the other Party shall be discharged from any liability.",
        metadata={"edition": "2017", "clause_number": "Sub-Clause 20.2.1", "clause_title": "Notice of Claim (28-Day Reciprocal Time-Bar)"}
    ),
    Document(
        page_content="FIDIC 2017 Red Book Sub-Clause 20.2.4 (Fully Detailed Claim): The claiming Party shall submit to the Engineer a fully detailed Claim within 84 days after the claiming Party became aware, or should have become aware, of the event or circumstance. The fully detailed Claim shall include a detailed description of the event, contemporaneous records, and contractual basis.",
        metadata={"edition": "2017", "clause_number": "Sub-Clause 20.2.4", "clause_title": "Fully Detailed Claim (84 Days)"}
    ),
    Document(
        page_content="FIDIC 2017 Red Book Sub-Clause 3.7 (Agreement or Determination): When carrying out duties under Sub-Clause 3.7, the Engineer shall act neutrally between the Parties and shall not be deemed to act for the Employer. The Engineer shall give Notice of agreement or determination within 42 days. If the Engineer fails to issue a determination within 42 days, the claim is deemed rejected.",
        metadata={"edition": "2017", "clause_number": "Sub-Clause 3.7", "clause_title": "Agreement or Determination (Neutrality & 42 Days)"}
    ),
    Document(
        page_content="FIDIC 2017 Red Book Sub-Clause 8.5 (Extension of Time for Completion): The Contractor shall be entitled subject to Sub-Clause 20.2 to an Extension of Time for Completion if and to the extent that completion is or will be delayed by any of the causes: Variations, exceptionally adverse climatic conditions, unforeseeable physical conditions under Sub-Clause 4.12, or delays caused by the Employer. Float buffers and concurrent contractor delays shall be audited under critical path delay principles.",
        metadata={"edition": "2017", "clause_number": "Sub-Clause 8.5", "clause_title": "Extension of Time for Completion"}
    ),
    Document(
        page_content="FIDIC 2017 Red Book Sub-Clause 8.8 (Delay Damages): If the Contractor fails to comply with Sub-Clause 8.2 (Time for Completion), the Employer shall be entitled subject to Sub-Clause 20.2 to payment of Delay Damages by the Contractor. These delay damages shall be the daily sum stated in the Contract Data, subject to the maximum limit (cap) stated. The Employer is required to issue a Sub-Clause 20.2 Notice.",
        metadata={"edition": "2017", "clause_number": "Sub-Clause 8.8", "clause_title": "Delay Damages"}
    ),
    Document(
        page_content="FIDIC 2017 Red Book Sub-Clause 21.4 (Obtaining DAAB's Decision): If a dispute arises between the Parties, either Party may refer the dispute in writing to the standing Dispute Avoidance/Adjudication Board (DAAB). The DAAB shall give its decision within 84 days after receiving the reference. The decision shall be binding on both Parties who shall promptly give effect to it.",
        metadata={"edition": "2017", "clause_number": "Sub-Clause 21.4", "clause_title": "Obtaining DAAB's Decision (Standing DAAB)"}
    )
]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
print("Ingesting all 12 FIDIC clauses (1999 & 2017 editions)...")
vector_db = Chroma.from_documents(documents=all_clauses, embedding=embeddings, persist_directory=persist_directory)
print("Ingestion complete. ChromaDB successfully created with 1999 and 2017 clauses.")