import os
from datetime import date, datetime
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# =====================================================================
# SECTION 1: UI CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="FIDIC Claims Engine | RAG + Deterministic CPU + Multi-Agent",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 26px; font-weight: 700; color: #1B365D; margin-bottom: 2px; }
    .sub-title { font-size: 14px; color: #64748B; margin-bottom: 20px; }
    .verdict-box { background-color: #F8FAFC; border-left: 5px solid #1B365D; padding: 18px; border-radius: 4px; margin-top: 15px; }
    .contractor-box { background-color: #EFF6FF; border-left: 5px solid #2563EB; padding: 14px; border-radius: 4px; margin-bottom: 12px; }
    .employer-box { background-color: #FEF2F2; border-left: 5px solid #DC2626; padding: 14px; border-radius: 4px; margin-bottom: 12px; }
    .accord-box { background-color: #F0FDF4; border-left: 5px solid #16A34A; padding: 16px; border-radius: 4px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ FIDIC Enterprise Claims & Delay Engine (Red Book 1999)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Bridge 1: ChromaDB RAG Vector Gate &nbsp;|&nbsp; Bridge 2: Deterministic CPU Engine &nbsp;|&nbsp; Bridge 3: Multi-Agent Negotiation Sentry</div>', unsafe_allow_html=True)

# =====================================================================
# SECTION 2: SIDEBAR GOVERNANCE & FINANCIAL PARAMETERS
# =====================================================================
try:
    cloud_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    cloud_key = os.environ.get("GEMINI_API_KEY", "")

MAX_SESSION_RUNS = 3
if "adjudication_runs" not in st.session_state:
    st.session_state.adjudication_runs = 0

with st.sidebar:
    st.header("⚙️ Engine Parameters")
    st.markdown("**Vector Store:** ChromaDB (Persistent Disk)")
    st.markdown("**Embeddings:** `gemini-embedding-2`")
    st.markdown("**Cognitive Model:** `gemini-3.5-flash` (Temp 0.0)")
    st.markdown("**Relevance Gate:** `0.45` Cosine Similarity")
    st.markdown("**Calculation Layer:** Python CPU (Zero Drift)")
    st.markdown("**Bridge 3:** Adversarial Multi-Agent DAB")
    st.divider()

    custom_key = st.text_input("Enter Your Own Gemini API Key (Optional)", type="password")
    if custom_key:
        api_key = custom_key
        os.environ["GEMINI_API_KEY"] = custom_key
        st.success("Custom User API Key Active (Unlimited Runs)")
    elif cloud_key:
        api_key = cloud_key
        os.environ["GEMINI_API_KEY"] = cloud_key
        remaining = max(0, MAX_SESSION_RUNS - st.session_state.adjudication_runs)
        st.info(f"Host Demo Key Active: {remaining}/{MAX_SESSION_RUNS} runs left")
    else:
        api_key = ""
        st.warning("Please provide a Gemini API Key to run determinations.")

    st.header("Contract Financial Terms")
    contract_value = st.number_input("Contract Sum ($):", min_value=100000, value=25000000, step=500000)
    ld_daily_rate = st.number_input("Daily LD Rate under Cl. 8.7 ($/day):", min_value=1000, value=15000, step=1000)
    ld_cap_percent = st.number_input("LD Maximum Cap (% of Contract):", min_value=1.0, max_value=25.0, value=10.0, step=0.5)

# =====================================================================
# SECTION 3: BRIDGE 2 DETERMINISTIC CPU CALCULATION LAYER
# =====================================================================
def calculate_delay_and_cost(d_event, d_notice, claimed_days, avail_float, concurrent_delay, total_delay_days, daily_rate, contract_val, cap_pct):
    notice_delta = (d_notice - d_event).days
    is_time_barred = notice_delta > 28

    if is_time_barred:
        net_eot = 0
        rejection_reason = f"Time-barred under Sub-Clause 20.1 (Notice served at {notice_delta} days; exceeds 28-day condition precedent)."
    else:
        slippage = max(0, claimed_days - avail_float)
        net_eot = max(0, slippage - concurrent_delay)
        rejection_reason = f"Procedurally compliant: Notice served within {notice_delta} days."

    unexcused_delay = max(0, total_delay_days - net_eot)
    raw_ld = unexcused_delay * daily_rate
    ld_cap_amount = (cap_pct / 100.0) * contract_val
    liquidated_damages = min(raw_ld, ld_cap_amount)
    is_capped = raw_ld >= ld_cap_amount

    return {
        "notice_delta": notice_delta,
        "is_time_barred": is_time_barred,
        "net_eot": net_eot,
        "rejection_reason": rejection_reason,
        "unexcused_delay": unexcused_delay,
        "liquidated_damages": liquidated_damages,
        "ld_cap_amount": ld_cap_amount,
        "is_capped": is_capped
    }

# =====================================================================
# SECTION 4: DISPUTE CHRONOLOGY & METRIC AUDIT DASHBOARD
# =====================================================================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Claim Narrative & Schedule Chronology")
    claim_statement = st.text_area(
        "Contractor Dispute Submittal Narrative:",
        value="Contractor encountered unexpectedly hard granite rock during station excavation, delaying critical path structural works.",
        height=90
    )

    c1, c2 = st.columns(2)
    with c1:
        date_event = st.date_input("Event Inception Date:", value=date(2026, 1, 10))
    with c2:
        date_notice = st.date_input("Contractor Notice Date:", value=date(2026, 1, 28))

    c3, c4, c5 = st.columns(3)
    with c3:
        claimed_days = st.number_input("Claimed EoT (Days):", min_value=1, value=40)
    with c4:
        avail_float = st.number_input("Available Total Float (Days):", min_value=0, value=5)
    with c5:
        concurrent_delay = st.number_input("Concurrent Contractor Delay (Days):", min_value=0, value=7)

    total_project_delay = st.number_input("Total Project Delay Recorded (Days):", min_value=0, value=40)

with col_right:
    st.subheader("2. Bridge 2: Deterministic Delay & LD Audit")
    
    calc = calculate_delay_and_cost(
        date_event, date_notice, claimed_days, avail_float, concurrent_delay,
        total_project_delay, ld_daily_rate, contract_value, ld_cap_percent
    )

    m1, m2 = st.columns(2)
    m1.metric("Notice Delta (Cl. 20.1)", f"{calc['notice_delta']} Days", "Limit: 28 Days", delta_color="inverse" if calc['is_time_barred'] else "normal")
    m2.metric("Procedural Gate", "TIME-BARRED" if calc['is_time_barred'] else "TIMELY NOTICE", delta_color="inverse" if calc['is_time_barred'] else "normal")

    m3, m4 = st.columns(2)
    m3.metric("Net Defensible EoT (Cl. 8.4)", f"{calc['net_eot']} Days", f"Claimed: {claimed_days}d")
    m4.metric("Delay Damages (Cl. 8.7)", f"${calc['liquidated_damages']:,.0f}", f"Unexcused: {calc['unexcused_delay']}d", delta_color="inverse" if calc['liquidated_damages'] > 0 else "normal")

    if calc['is_time_barred']:
        st.error(f"❌ {calc['rejection_reason']}")
    else:
        st.success(f"✅ {calc['rejection_reason']}")

st.divider()

# =====================================================================
# SECTION 5: UNIFIED ADJUDICATION PIPELINE (BRIDGE 1 & BRIDGE 2)
# =====================================================================
is_limit_reached = (not custom_key) and (st.session_state.adjudication_runs >= MAX_SESSION_RUNS)

if is_limit_reached:
    st.warning("⚠️ Session demo limit reached (3/3 runs used). To run further adjudications, enter your own Gemini API Key in the left sidebar.")

col_btn1, col_btn2 = st.columns(2)
run_btn_single = col_btn1.button("⚖️ Run Engineer's Determination (Sub-Cl. 3.5)", type="primary", disabled=is_limit_reached)
run_btn_multi = col_btn2.button("🤝 Run Bridge 3: Multi-Agent DAB Negotiation", disabled=is_limit_reached)

def get_rag_context(statement):
    persist_dir = "./chroma_fidic_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    results = vector_db.similarity_search_with_relevance_scores(statement, k=3)
    retrieved = [doc for doc, score in results if score >= 0.45]
    context = "\n\n".join([
        f"--- {doc.metadata.get('clause_number', 'Clause')}: {doc.metadata.get('clause_title', 'Title')} ---\n{doc.page_content}"
        for doc in retrieved
    ])
    return results, retrieved, context

if run_btn_single:
    if not api_key:
        st.error("API Key missing. Please provide a Gemini API Key.")
        st.stop()

    with st.spinner("Retrieving verified clauses and synthesizing formal determination..."):
        try:
            results, retrieved_clauses, context_text = get_rag_context(claim_statement)

            st.subheader("Step 1: Contractual Relevance Audit (Bridge 1)")
            cols = st.columns(3)
            for idx, (doc, score) in enumerate(results):
                passed = score >= 0.45
                clause_tag = doc.metadata.get("clause_number", doc.metadata.get("clause", "Clause"))
                title_tag = doc.metadata.get("clause_title", doc.metadata.get("title", "Condition"))
                with cols[idx]:
                    st.metric(label=clause_tag, value=f"{score:.4f}", delta="Passed Gate" if passed else "Rejected", delta_color="normal" if passed else "inverse")
                    st.caption(f"**{title_tag}**")
                    st.text(doc.page_content[:140] + "...")

            if not retrieved_clauses:
                st.error("Relevance Gate Audit: No contract clauses exceeded 0.45 threshold. Claim dismissed.")
                st.stop()

            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
            system_prompt = f"""You are an impartial FIDIC Engineer operating strictly under FIDIC 1999 Red Book rules.

VERIFIED CONTRACT EXCERPTS (Bridge 1 RAG):
{context_text}

DETERMINISTIC CPU AUDIT METRICS (Bridge 2 Mathematical Facts - DO NOT ALTER):
- Notice Delta: {calc['notice_delta']} days (Limit: 28 days | Status: {'TIME-BARRED' if calc['is_time_barred'] else 'TIMELY'})
- Claimed EoT: {claimed_days} days | Float Consumed: {avail_float} days | Concurrent Delay: {concurrent_delay} days
- Net Defensible EoT: {calc['net_eot']} days | Unexcused Delay: {calc['unexcused_delay']} days
- Liquidated Delay Damages Assessed (Cl. 8.7): ${calc['liquidated_damages']:,.2f}

DISPUTE NARRATIVE:
{claim_statement}

Draft a formal Engineer's Determination Letter under Sub-Clause 3.5:
1. CONTRACTUAL CLAUSES INVOKED
2. PROCEDURAL TIME-BAR DETERMINATION (Sub-Clause 20.1)
3. EXTENSION OF TIME & CONCURRENCY ANALYSIS (SCL Float Offset)
4. FINANCIAL RULING & DELAY DAMAGES (Sub-Clause 8.7)
"""
            response = llm.invoke(system_prompt)
            output_text = response.content[0].get("text", "") if isinstance(response.content, list) else response.content

            if not custom_key:
                st.session_state.adjudication_runs += 1

            st.subheader("Step 2: Formal Engineer's Determination Memorandum")
            st.markdown(f'<div class="verdict-box">{output_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Execution Error: {e}")

# =====================================================================
# SECTION 6: BRIDGE 3 - AUTONOMOUS MULTI-AGENT NEGOTIATION SENTRY
# =====================================================================
if run_btn_multi:
    if not api_key:
        st.error("API Key missing. Please provide a Gemini API Key.")
        st.stop()

    with st.spinner("Deploying Contractor Advocate & Employer DAB Sentry Agents..."):
        try:
            results, retrieved_clauses, context_text = get_rag_context(claim_statement)
            if not retrieved_clauses:
                st.error("Relevance Gate Audit: No verified contract clauses found. Negotiation aborted.")
                st.stop()

            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

            # Agent 1: Contractor Advocate
            prompt_contractor = f"""You are the Lead Claims Advocate representing the EPC Contractor under FIDIC 1999 Red Book.
CONTRACT CONTEXT: {context_text}
AUDITED FACTS: Event Inception: {date_event}, Notice Served: {date_notice} ({calc['notice_delta']} days). Claimed EoT: {claimed_days} days. Total project delay: {total_project_delay} days.
DISPUTE NARRATIVE: {claim_statement}

Your Objective: Fight aggressively for full {claimed_days} days EoT and full contractor relief.
Argue why the notice was timely, why float belongs to the contractor, challenge concurrent delays, and reject any liquidated damages. Be assertive, legally structured, and concise (under 200 words)."""
            
            res_contractor = llm.invoke(prompt_contractor)
            contractor_speech = res_contractor.content[0].get("text", "") if isinstance(res_contractor.content, list) else res_contractor.content

            # Agent 2: Employer Dispute Board (DAB) Sentry
            prompt_employer = f"""You are the Dispute Adjudication Board (DAB) Legal Sentry defending the Employer under FIDIC 1999 Red Book.
CONTRACT CONTEXT: {context_text}
DETERMINISTIC CPU AUDITED METRICS:
- Notice Delta: {calc['notice_delta']} days (Ceiling: 28 days | Time-Bar: {calc['is_time_barred']})
- SCL Float Available: {avail_float} days consumed
- Concurrent Contractor Delay: {concurrent_delay} days deducted
- Net Defensible EoT: {calc['net_eot']} days
- Unexcused Delay: {calc['unexcused_delay']} days
- Contractual Delay Damages Levied: ${calc['liquidated_damages']:,.2f}

CONTRACTOR ADVOCATE'S ARGUMENT:
{contractor_speech}

Your Objective: Strictly enforce contractual conditions precedent. If notice > 28 days, enforce Sub-Clause 20.1 time-bar ruthlessly. Enforce SCL concurrency offsets and demand full recovery of ${calc['liquidated_damages']:,.2f} in liquidated damages under Sub-Clause 8.7. Be decisive and concise (under 200 words)."""

            res_employer = llm.invoke(prompt_employer)
            employer_speech = res_employer.content[0].get("text", "") if isinstance(res_employer.content, list) else res_employer.content

            # Agent 3: Mediated Settlement Accord
            prompt_accord = f"""You are an accredited Senior Arbitrator / FIDIC Adjudicator presiding over this dispute.
FACTUAL DETERMINISTIC CPU METRICS: Net EoT: {calc['net_eot']} days. Liquidated Damages: ${calc['liquidated_damages']:,.2f}. Notice Delta: {calc['notice_delta']} days.
CONTRACTOR SUBMISSION: {contractor_speech}
EMPLOYER DAB REBUTTAL: {employer_speech}

Synthesize a binding 'Dispute Adjudication Board Settlement Accord' under Sub-Clause 20.4.
Provide:
1. FINAL BINDING EOT AWARD (Days)
2. FINAL ENFORCEABLE DELAY DAMAGES ($)
3. 3-POINT COMMERCIAL SETTLEMENT ACTION PLAN (Clear bullets for site execution)"""

            res_accord = llm.invoke(prompt_accord)
            accord_speech = res_accord.content[0].get("text", "") if isinstance(res_accord.content, list) else res_accord.content

            if not custom_key:
                st.session_state.adjudication_runs += 1

            st.subheader("Step 3: Multi-Agent Adversarial Negotiation Transcript")
            tab_view, tab_raw = st.tabs(["🏛️ Adversarial Debate Stream", "📜 Settlement Term Sheet"])

            with tab_view:
                st.markdown(f'<div class="contractor-box"><b>👷 Contractor Advocate Agent:</b><br>{contractor_speech.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="employer-box"><b>🏛️ Employer DAB Sentry Agent:</b><br>{employer_speech.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            with tab_raw:
                st.markdown(f'<div class="accord-box"><b>⚖️ DAB Mediated Settlement Accord (Sub-Clause 20.4):</b><br>{accord_speech.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Negotiation Execution Error: {e}")