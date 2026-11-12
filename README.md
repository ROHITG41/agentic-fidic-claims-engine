# Autonomous FIDIC Contract Claims & Time-Bar Engine

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rohitkumarg-fidic-engine.streamlit.app)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Vector DB](https://img.shields.io/badge/ChromaDB-Persistent%20Disk-D97706)
![Cognitive LLM](https://img.shields.io/badge/Cognitive%20LLM-gemini--3.5--flash-7C3AED)
![Architecture](https://img.shields.io/badge/Architecture-Autonomous%20Multi--Agent-059669)

> **Dataset & Confidentiality Note:** This application is an independent engineering architecture demonstration validated on representative, anonymized project datasets.

---

### Executive Problem Statement
Under FIDIC 1999/2017 Red and Yellow Book contracts, strict procedural conditions precedent—most notably the **28-day notice time-bar under Sub-Clause 20.1 / 20.2**—determine whether a contractor's claim is valid or time-barred. Manual cross-referencing of notice chronologies against thousands of pages of site correspondence and baseline logs often results in missed procedural defenses or costly arbitration.

### Operational Value
* **Early Procedural Defense:** Deterministically audits event awareness dates against formal notice dates to flag fatal 28-day time-bar defaults immediately.
* **Evidence-Grounded Rebuttals:** Retrieves contemporaneous site records and contract clauses via ChromaDB Vector RAG to prevent unsupported LLM citations.
* **Structured Determination Drafting:** Models Dispute Avoidance/Adjudication Board (DAAB) positions and drafts formal Engineer's Determinations under Sub-Clause 3.5 / 3.7 for human review.

---

### System Architecture

```mermaid
flowchart TD
    classDef client fill:#EFF6FF,stroke:#2563EB,stroke-width:2px,color:#1E3A8A;
    classDef deterministic fill:#ECFDF5,stroke:#059669,stroke-width:2px,color:#064E3B;
    classDef memory fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#78350F;
    classDef logic fill:#FAF5FF,stroke:#7C3AED,stroke-width:2px,color:#4C1D95;
    classDef output fill:#F8FAFC,stroke:#475569,stroke-width:2px,color:#0F172A;

    subgraph INGRESS ["1. CLAIM CHRONOLOGY & NOTICE INGESTION"]
        UI["Streamlit Cloud Interface<br/><b>Event Trigger Date, Notice Date & Claim Particulars</b>"]:::client
    end

    subgraph CPU_GATE ["2. DETERMINISTIC TIME-BAR & QUANTUM GATE"]
        CAL["<b>Statutory Calendar Sentry (CPU)</b><br/>Computes Exact Elapsed Days vs. 28-Day FIDIC Cl. 20.1 Gate<br/>Validates 42-Day Detailed Particulars Window"]:::deterministic
    end

    subgraph RAG_STORE ["3. SEMANTIC EVIDENCE RETRIEVAL"]
        CHROMA[("<b>ChromaDB Vector Store</b><br/>FIDIC Contract Clauses & Contemporaneous Site Logs<br/>gemini-embedding-2 Semantic Search")]:::memory
    end

    subgraph COGNITIVE ["4. MULTI-AGENT ADJUDICATION & DAAB SIMULATION"]
        LLM["<b>LangChain LCEL + Google Gemini (gemini-3.5-flash)</b><br/>Synthesizes Engineer's Rebuttal (Cl. 3.5 / 3.7)<br/>Models Adversarial DAAB Settlement Scenarios"]:::logic
    end

    subgraph DELIVERABLE ["5. CONTRACTUAL DETERMINATION ARTIFACT"]
        MEMO["<b>Formal Engineer's Determination & Rebuttal Dossier</b><br/>Downloadable Memorandum for Commercial Review"]:::output
    end

    UI --> CAL
    UI --> CHROMA
    CAL -->|Procedural Time-Bar Verdict| LLM
    CHROMA -->|Grounded Evidence Chunks| LLM
    LLM --> MEMO
```

---

### Technology Stack
* **Runtime & UI:** Python 3.12, Streamlit Community Cloud
* **Vector Database & Retrieval:** ChromaDB, Google `gemini-embedding-2`
* **Orchestration & Validation:** LangChain LCEL, Google `gemini-3.5-flash`, Pydantic v2

### Author & Architect
**Rohit Kumar, PMP, CEng, CEA, FSEr (TÜV)**  
Cross-Technology Technical Architect | Mission-Critical Infrastructure & Applied AI Solutions  
[LinkedIn](https://www.linkedin.com/in/rohitg41) | [Live Application](https://rohitkumarg-fidic-engine.streamlit.app)

---

### Enterprise Advisory & Custom Architecture
Available for technical consulting, custom agentic workflow development, and private enterprise deployments.  
**Connect:** [LinkedIn](https://www.linkedin.com/in/rohitg41) | **Email:** `grohit41@gmail.com`

---

### Intellectual Property & Licensing
**Copyright © 2026 Rohit Kumar. All Rights Reserved.**  
This repository and its source code are published solely for technical portfolio evaluation and architectural demonstration. Unauthorized copying, redistribution, or commercial deployment is prohibited without prior written consent.
