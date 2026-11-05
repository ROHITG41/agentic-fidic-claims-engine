# ⚖️ FIDIC Enterprise Claims Engine (Red Book 1999)

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-fidic-claims-engine.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Vector Store](https://img.shields.io/badge/ChromaDB-Persistent%20Disk-red)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Google-gemini--embedding--2-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Reasoning Engine](https://img.shields.io/badge/Cognitive%20LLM-gemini--3.5--flash%20(Temp%200.0)-8E75C2)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for deterministically adjudicating construction contract disputes under the **FIDIC 1999 Conditions of Contract for Construction (Red Book)**. 

Designed for Contract Administrators, Claims Consultants, and Legal Counsel, the engine mitigates subjective hallucination by enforcing an on-disk vector index, an empirical relevance rejection gate, and zero-temperature legal grounding.

---

## 🏛️ Business & Legal Problem Statement

FIDIC contracts govern multi-million and multi-billion-dollar infrastructure developments. Mismanagement of formal claim procedures introduces critical enterprise liabilities:
* **Sub-Clause 20.1 Fatal Time-Bars:** If a contractor fails to give notice of a claim within 28 days of becoming aware of the event, the entitlement is extinguished and the Employer is discharged from all liability.
* **Sub-Clause 4.12 Unforeseeable Physical Conditions:** Strict evidentiary burdens determine whether subsurface obstructions qualify as unforeseeable to an experienced contractor.
* **Sub-Clause 8.4 Extension of Time (EoT):** Entitlement requires direct causal linkage between the delay event and the critical path of the Works.

Conventional conversational LLMs frequently hallucinate provisions, cite outdated editions, or overlook procedural time-bars. This engine eliminates non-deterministic variance by chaining dense semantic retrieval with strict contractual boundary gates.

---

## 📐 System Architecture

```
Contractor Dispute Submittal
             │
             ▼
   [gemini-embedding-2]
   Dense Vector Generation
             │
             ▼
    [ChromaDB Local Store]
  Top-k Cosine Vector Query (k=3)
             │
             ▼
┌──────────────────────────────┐
│    Calibrated Relevance      │
│     Threshold Audit          ├────── [Score < 0.45] ──► Rejection Gate
│    (Cosine Sim >= 0.45)      │                         (No Extraneous LLM Call)
└────────────┬─────────────────┘
             │ [Score >= 0.45]
             ▼
  Grounded Context Injection
  (FIDIC 1999 Contract Excerpts)
             │
             ▼
    [gemini-3.5-flash]
   Temperature: 0.0 (Greedy)
             │
             ▼
Formal Legal Adjudication Memorandum
• Governing Sub-Clauses Retrieved
• Entitlement Verdict (TIME-BARRED / ENTITLED / REJECTED)
• Impartial Contractual Rationale
```

---

## 🔬 Core Engineering Innovations

| Feature | Technical Implementation | Enterprise Impact |
| :--- | :--- | :--- |
| **Persistent Vector Store** | Embedded on-disk ChromaDB (`./chroma_fidic_db`) | Sub-100ms retrieval without re-indexing overhead or vector recalculation. |
| **Vector Geometry** | Google `gemini-embedding-2` dense coordinates | Captures deep semantic nuances across legal phrasing and claim narratives. |
| **Calibrated Relevance Gate** | Cutoff threshold set at `0.45` cosine similarity | Rejects out-of-scope or irrelevant queries before hitting the inference model. |
| **Deterministic Adjudication** | `gemini-3.5-flash` at `temperature=0.0` | Eliminates creative drift; rulings are strictly reproducible and grounded. |
| **Production UI** | Streamlit Cloud deployment with Secrets injection | Interactive parameter auditing and real-time retrieval metric verification. |

---

## 📊 Empirical Verification & Test Matrix

The engine comes pre-calibrated against two core contractual scenarios:

### Case 1: Procedural Time-Bar (Sub-Clause 20.1)
* **Input:** `"Contractor submitted formal delay notice 35 days after flood event"`
* **ChromaDB Top Match:** Sub-Clause 20.1 (Relevance: `~0.65` — **Passed Gate**)
* **Adjudication Verdict:** **TIME-BARRED**
* **Rationale:** The 35-day submittal interval breaches the 28-day strict statutory ceiling established in Sub-Clause 20.1 [Contractor's Claims], discharging the Employer from liability.

### Case 2: Unforeseeable Physical Conditions (Sub-Clause 4.12 & 8.4)
* **Input:** `"Contractor encountered unexpectedly hard granite rock during station excavation"`
* **ChromaDB Top Matches:** Sub-Clause 4.12 (`0.5225`), Sub-Clause 8.4 (`0.4753`) — **Passed Gate**
* **Adjudication Verdict:** **ENTITLED TO EXTENSION OF TIME & COST**
* **Rationale:** Subsurface granite falls within the scope of physical conditions defined in Sub-Clause 4.12, establishing causal ground for an Extension of Time pursuant to Sub-Clause 8.4.

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/ROHITG41/agentic-fidic-claims-engine.git](https://github.com/ROHITG41/agentic-fidic-claims-engine.git)
cd agentic-fidic-claims-engine
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment & Run
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY="your_gemini_api_key"
streamlit run app.py

# Linux/macOS:
export GEMINI_API_KEY="your_gemini_api_key"
streamlit run app.py
```

---

## 🗺️ Roadmap & Multi-Agent Bridge Architecture

* [x] **Bridge 1 (Completed):** RAG Ingestion Pipeline, Persistent ChromaDB Storage, Calibrated Gate, Streamlit Cloud Deployment.
* [ ] **Bridge 2:** Deterministic Delay & Cost Engine — Programmatic CPM delay analysis and Liquidated Damages arithmetic.
* [ ] **Bridge 3:** Autonomous Multi-Agent Negotiation Sentry — Contractor Advocate Agent vs. Employer Dispute Board Agent.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.