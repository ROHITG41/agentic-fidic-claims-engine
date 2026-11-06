# ⚖️ FIDIC Enterprise Claims & Delay Engine (Red Book 1999)

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rohit-kumar-fidic-claims.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Vector Store](https://img.shields.io/badge/ChromaDB-Persistent%20Disk-red)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Google-gemini--embedding--2-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Reasoning Engine](https://img.shields.io/badge/Cognitive%20LLM-gemini--3.5--flash%20(Temp%200.0)-8E75C2)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, hybrid AI adjudication system engineered for construction disputes under the FIDIC 1999 Conditions of Contract for Construction (Red Book). 

The platform eliminates AI hallucination through a dual-engine architecture: an on-disk ChromaDB semantic vector gate (Bridge 1) coupled with a deterministic CPU calculation layer (Bridge 2) that programmatically enforces contractual time-bars, float consumption, and liquidated damages caps.

---

## 🏛️ Business & Legal Problem Statement

FIDIC contracts govern multi-million and multi-billion-dollar infrastructure developments. Traditional conversational LLMs present fatal risks in contract administration: they hallucinate clauses, cite incorrect editions, miscalculate calendar day deltas, and make rounding errors on delay damages.

This engine establishes strict deterministic boundaries:
* **Sub-Clause 20.1 Fatal Time-Bars:** Failure to submit a notice within 28 days of event awareness forfeits all entitlement to time and cost, discharging the Employer from liability.
* **Sub-Clause 4.12 Unforeseeable Physical Conditions:** Enforces rigorous evidentiary thresholds to determine if subsurface obstructions qualify for relief.
* **Sub-Clause 8.4 Critical Path Entitlement:** Programmatically deducts available buffer float and contractor concurrent delays under SCL Delay Protocol principles.
* **Sub-Clause 8.7 Delay Damages Math:** Liquidated damages are calculated strictly on unexcused delay days and capped at the agreed contractual ceiling.

---

## 📐 System Architecture: Hybrid RAG + Deterministic CPU

```text
Contractor Claim Narrative & Schedule Chronology
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[Bridge 1: Legal Vector Gate]  [Bridge 2: Deterministic CPU Layer]
ChromaDB + gemini-embedding-2   Python Date & Float Arithmetic
Relevance Gate Cutoff >= 0.45   Zero-Hallucination Math Execution
        │                           │
        │ Verified Clauses          │ Audited Days, EoT & LDs ($)
        └─────────────┬─────────────┘
                      ▼
            [gemini-3.5-flash]
       Greedy Decoding @ Temperature 0.0
                      │
                      ▼
    Formal Engineer's Determination Letter
    • Sub-Clause 20.1 Procedural Ruling
    • Sub-Clause 8.4 Net Defensible EoT
    • Sub-Clause 8.7 Enforceable Delay Damages ($)
```

---

## 🔬 Core Engineering Innovations

| Layer | Technical Implementation | Enterprise Safeguard |
| :--- | :--- | :--- |
| **Bridge 1: Persistent Vector Store** | Embedded on-disk ChromaDB (`./chroma_fidic_db`) | Sub-100ms retrieval without recurring re-indexing latency. |
| **Bridge 1: Relevance Guardrail** | Cosine similarity cutoff set at `0.45` | Rejects non-contractual queries prior to model inference. |
| **Bridge 2: Date Delta Engine** | Pure Python calendar delta calculation | Enforces strict 28-day Sub-Clause 20.1 time-bar condition precedent. |
| **Bridge 2: Delay & Cost Engine** | SCL float offsets & Sub-Clause 8.7 LD formulas | Guarantees exact liquidated damages math capped at contractual limits. |
| **Cognitive Adjudicator** | `gemini-3.5-flash` at `temperature=0.0` | Eliminates creative drift; rulings are grounded strictly on retrieved facts. |

---

## 📐 Bridge 2 Deterministic Formulas

1. **Sub-Clause 20.1 Statutory Notice Delta:**
   $$\Delta_{\text{notice}} = \text{Date}_{\text{Notice}} - \text{Date}_{\text{Awareness}}$$
   $$\text{If } \Delta_{\text{notice}} > 28\text{ days} \implies \text{Status: TIME-BARRED (0 Days Entitlement)}$$

2. **Sub-Clause 8.4 Critical Path Float & Concurrency Offset:**
   $$\text{Critical Slippage} = \max(0, \text{Claimed Days} - \text{Total Float Consumed})$$
   $$\text{Net Defensible EoT} = \max(0, \text{Critical Slippage} - \text{Concurrent Contractor Delay})$$

3. **Sub-Clause 8.7 Delay Damages Exposure:**
   $$\text{Unexcused Delay} = \max(0, \text{Total Project Delay} - \text{Net Defensible EoT})$$
   $$\text{Damages Levied} = \min(\text{Unexcused Delay} \times \text{Daily Rate}, \text{Contractual Cap})$$

---

## 📊 Empirical Verification & Test Matrix

### Case 1: Procedural Time-Bar (Sub-Clause 20.1)
* **Scenario:** Notice served 35 days after event awareness (Limit: 28 days).
* **Deterministic CPU Output:** `Notice Delta: 35 days` | `Status: TIME-BARRED` | `Net EoT: 0 days`.
* **Adjudication Verdict:** **TIME-BARRED**. Employer fully discharged from liability.

### Case 2: Unforeseeable Physical Conditions & Delay Damages (Sub-Clause 4.12, 8.4, 8.7)
* **Scenario:** Contractor encounters unforeseen granite rock. 40 days claimed, 5 days float available, 7 days concurrent contractor delay, 40 days total project delay. Contract: $25M, LD rate: $15,000/day.
* **Deterministic CPU Output:**
  * Net Defensible EoT: $40 - 5 - 7 = \mathbf{28\text{ days}}$.
  * Unexcused Culpable Delay: $40 - 28 = \mathbf{12\text{ days}}$.
  * Delay Damages Levied: $12 \times \$15,000 = \mathbf{\$180,000}$.
* **Adjudication Verdict:** **ADMITTED IN PART (28 Days EoT Awarded, $180,000 LDs Assessed)**.

---

## 🚀 Quickstart & Local Installation

```bash
# 1. Clone repository
git clone [https://github.com/ROHITG41/agentic-fidic-claims-engine.git](https://github.com/ROHITG41/agentic-fidic-claims-engine.git)
cd agentic-fidic-claims-engine

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app_fidic_claims.py
```

---

## 🗺️ Multi-Agent Bridge Roadmap

- [x] **Bridge 1 (Completed):** RAG Ingestion Pipeline, Persistent ChromaDB Storage, 0.45 Relevance Gate, Streamlit Cloud Deployment.
- [x] **Bridge 2 (Completed):** Deterministic Delay & Cost Engine — Zero-drift date arithmetic, SCL float deduction, and Sub-Clause 8.7 damages calculation.
- [ ] **Bridge 3 (Next Milestone):** Autonomous Multi-Agent Negotiation Sentry — Contractor Advocate Agent vs. Employer Dispute Board Agent.

---

## 📄 License
Distributed under the MIT License.