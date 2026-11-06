import os
import io
from datetime import date, datetime
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# =====================================================================
# SECTION 1: UI CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="FIDIC Claims Engine | Multi-Edition (1999 & 2017)",
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

# =====================================================================
# SECTION 2: SIDEBAR GOVERNANCE & MULTI-EDITION MAPPING
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
    
    # Contract Edition Toggle
    selected_edition = st.selectbox(
        "Select FIDIC Edition:",
        ["FIDIC 1999 Red Book", "FIDIC 2017 Second Edition"],
        index=0
    )
    is_2017 = selected_edition == "FIDIC 2017 Second Edition"

    st.markdown("**Vector Store:** ChromaDB (Multi-Edition)")
    st.markdown("**Embeddings:** `gemini-embedding-2`")
    st.markdown("**Cognitive Model:** `gemini-3.5-flash` (Temp 0.0)")
    st.markdown("**Calculation Layer:** Python CPU (Zero Drift)")
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
    ld_daily_rate = st.number_input("Daily LD Rate ($/day):", min_value=1000, value=15000, step=1000)
    ld_cap_percent = st.number_input("LD Maximum Cap (% of Contract):", min_value=1.0, max_value=25.0, value=10.0, step=0.5)

# Contract Clause Mapping
clause_map = {
    "notice": "Sub-Clause 20.2.1" if is_2017 else "Sub-Clause 20.1",
    "notice_desc": "28-Day Reciprocal Condition Precedent" if is_2017 else "28-Day Condition Precedent",
    "eot": "Sub-Clause 8.5" if is_2017 else "Sub-Clause 8.4",
    "ld": "Sub-Clause 8.8" if is_2017 else "Sub-Clause 8.7",
    "determination": "Sub-Clause 3.7 (Neutrality & 42 Days)" if is_2017 else "Sub-Clause 3.5 (Engineer's Determination)",
    "board": "Standing DAAB (Sub-Clause 21.4)" if is_2017 else "Ad-hoc DAB (Sub-Clause 20.4)"
}

st.markdown(f'<div class="main-title">⚖️ FIDIC Enterprise Claims & Delay Engine ({selected_edition})</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Bridge 1: ChromaDB RAG &nbsp;|&nbsp; Bridge 2: Deterministic CPU ({clause_map["notice"]}) &nbsp;|&nbsp; Bridge 3: Multi-Agent {clause_map["board"]}</div>', unsafe_allow_html=True)

# =====================================================================
# SECTION 3: BRIDGE 2 DETERMINISTIC CPU CALCULATION LAYER
# =====================================================================
def calculate_delay_and_cost(d_event, d_notice, claimed_days, avail_float, concurrent_delay, total_delay_days, daily_rate, contract_val, cap_pct, notice_clause):
    notice_delta = (d_notice - d_event).days
    is_time_barred = notice_delta > 28

    if is_time_barred:
        net_eot = 0
        rejection_reason = f"Time-barred under {notice_clause} (Notice served at {notice_delta} days; exceeds 28-day statutory condition precedent)."
    else:
        slippage = max(0, claimed_days - avail_float)
        net_eot = max(0, slippage - concurrent_delay)
        rejection_reason = f"Procedurally compliant under {notice_clause}: Notice served within {notice_delta} days."

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
# SECTION 4: PDF GENERATION ENGINE
# =====================================================================
def clean_pdf_text(text):
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2022': '*', '•': '*',
        '⚖️': '', '✅': '[PASS]', '❌': '[FAIL]', '🔹': '-',
        '🏛️': '', '👷': '', '🤝': '', '📊': ''
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(edition_title, report_title, narrative, event_d, notice_d, claimed_d, calc_data, decision_text, cmap):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('RepTitle', parent=styles['Heading1'], fontSize=15, leading=19, textColor=colors.HexColor('#1B365D'))
    h2_style = ParagraphStyle('RepH2', parent=styles['Heading2'], fontSize=11, leading=15, textColor=colors.HexColor('#1B365D'))
    body_style = ParagraphStyle('RepBody', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#1E293B'))

    elements = []
    elements.append(Paragraph(clean_pdf_text(f"<b>{report_title}</b>"), title_style))
    elements.append(Paragraph(clean_pdf_text(f"Governing Standard: {edition_title} | Adjudication Record"), body_style))
    elements.append(Spacer(1, 12))

    status_label = "TIME-BARRED" if calc_data['is_time_barred'] else "TIMELY NOTICE"
    table_data = [
        ["Parameter", "Audited Metric", "Contractual Governing Basis"],
        [f"Notice Delta ({cmap['notice']})", f"{calc_data['notice_delta']} Days", f"Limit: 28 Days ({status_label})"],
        ["Extension of Time Claimed", f"{claimed_d} Days", "Contractor Submission"],
        [f"Net Defensible EoT ({cmap['eot']})", f"{calc_data['net_eot']} Days", "SCL Protocol Concurrency & Float Deduction"],
        ["Unexcused Delay Assessed", f"{calc_data['unexcused_delay']} Days", f"Subject to {cmap['ld']} Delay Damages"],
        ["Liquidated Damages Levied", f"${calc_data['liquidated_damages']:,.2f}", f"Contractual Cap Applied: {'Yes' if calc_data['is_capped'] else 'No'}"]
    ]

    t = Table(table_data, colWidths=[155, 110, 245])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B365D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("<b>1. Claim Particulars & Factual Submittal</b>", h2_style))
    elements.append(Paragraph(clean_pdf_text(narrative), body_style))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>2. Binding Adjudication Ruling</b>", h2_style))
    for paragraph in decision_text.split("\n"):
        if paragraph.strip():
            elements.append(Paragraph(clean_pdf_text(paragraph), body_style))
            elements.append(Spacer(1, 4))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================================
# SECTION 5: DISPUTE CHRONOLOGY & METRIC AUDIT DASHBOARD
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
    st.subheader(f"2. Bridge 2: Deterministic Delay & LD Audit ({clause_map['notice']})")
    
    calc = calculate_delay_and_cost(
        date_event, date_notice, claimed_days, avail_float, concurrent_delay,
        total_project_delay, ld_daily_rate, contract_value, ld_cap_percent, clause_map["notice"]
    )

    m1, m2 = st.columns(2)
    m1.metric(f"Notice Delta ({clause_map['notice']})", f"{calc['notice_delta']} Days", "Limit: 28 Days", delta_color="inverse" if calc['is_time_barred'] else "normal")
    m2.metric("Procedural Gate", "TIME-BARRED" if calc['is_time_barred'] else "TIMELY NOTICE", delta_color="inverse" if calc['is_time_barred'] else "normal")

    m3, m4 = st.columns(2)
    m3.metric(f"Net Defensible EoT ({clause_map['eot']})", f"{calc['net_eot']} Days", f"Claimed: {claimed_days}d")
    m4.metric(f"Delay Damages ({clause_map['ld']})", f"${calc['liquidated_damages']:,.0f}", f"Unexcused: {calc['unexcused_delay']}d", delta_color="inverse" if calc['liquidated_damages'] > 0 else "normal")

    if calc['is_time_barred']:
        st.error(f"❌ {calc['rejection_reason']}")
    else:
        st.success(f"✅ {calc['rejection_reason']}")

st.divider()

# =====================================================================
# SECTION 6: UNIFIED ADJUDICATION PIPELINE & MULTI-AGENT BOARD
# =====================================================================
is_limit_reached = (not custom_key) and (st.session_state.adjudication_runs >= MAX_SESSION_RUNS)

if is_limit_reached:
    st.warning("⚠️ Session demo limit reached (3/3 runs used). To run further adjudications, enter your own Gemini API Key in the left sidebar.")

col_btn1, col_btn2 = st.columns(2)
run_btn_single = col_btn1.button(f"⚖️ Run Determination ({clause_map['determination']})", type="primary", disabled=is_limit_reached)
run_btn_multi = col_btn2.button(f"🤝 Run Bridge 3: Multi-Agent Negotiation ({clause_map['board']})", disabled=is_limit_reached)

def get_rag_context(statement, edition_tag):
    persist_dir = "./chroma_fidic_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Filter search by edition
    filter_dict = {"edition": "2017"} if edition_tag else None
    results = vector_db.similarity_search_with_relevance_scores(statement, k=4, filter=filter_dict)
    
    retrieved = [doc for doc, score in results if score >= 0.40]
    context = "\n\n".join([
        f"--- {doc.metadata.get('clause_number', 'Clause')}: {doc.metadata.get('clause_title', 'Title')} ---\n{doc.page_content}"
        for doc in retrieved
    ])
    return results, retrieved, context

# Path A: Determination
if run_btn_single:
    if not api_key:
        st.error("API Key missing. Please provide a Gemini API Key.")
        st.stop()

    with st.spinner(f"Auditing {selected_edition} conditions and synthesizing determination..."):
        try:
            results, retrieved_clauses, context_text = get_rag_context(claim_statement, is_2017)

            st.subheader(f"Step 1: Contractual Relevance Audit ({selected_edition})")
            cols = st.columns(len(results))
            for idx, (doc, score) in enumerate(results):
                passed = score >= 0.40
                clause_tag = doc.metadata.get("clause_number", "Clause")
                title_tag = doc.metadata.get("clause_title", "Condition")
                with cols[idx]:
                    st.metric(label=clause_tag, value=f"{score:.4f}", delta="Passed Gate" if passed else "Rejected", delta_color="normal" if passed else "inverse")
                    st.caption(f"**{title_tag}**")
                    st.text(doc.page_content[:120] + "...")

            if not retrieved_clauses:
                st.error("Relevance Gate Audit: No relevant contract clauses passed threshold. Claim dismissed.")
                st.stop()

            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
            system_prompt = f"""You are an impartial FIDIC Engineer operating strictly under {selected_edition} rules.

VERIFIED CONTRACT EXCERPTS (Bridge 1 RAG):
{context_text}

DETERMINISTIC CPU AUDIT METRICS (Bridge 2 Mathematical Facts - DO NOT ALTER):
- Notice Delta: {calc['notice_delta']} days (Limit: 28 days | Status: {'TIME-BARRED' if calc['is_time_barred'] else 'TIMELY'})
- Notice Governing Clause: {clause_map['notice']} ({clause_map['notice_desc']})
- Claimed EoT: {claimed_days} days | Float Consumed: {avail_float} days | Concurrent Delay: {concurrent_delay} days
- Net Defensible EoT: {calc['net_eot']} days under {clause_map['eot']} | Unexcused Delay: {calc['unexcused_delay']} days
- Liquidated Delay Damages Assessed: ${calc['liquidated_damages']:,.2f} under {clause_map['ld']}

DISPUTE NARRATIVE:
{claim_statement}

Draft a formal Engineer's Determination Letter under {clause_map['determination']}:
1. CONTRACTUAL CLAUSES INVOKED (Cite exact {selected_edition} Sub-Clauses)
2. PROCEDURAL TIME-BAR DETERMINATION (Strict ruling under {clause_map['notice']})
3. EXTENSION OF TIME & CONCURRENCY ANALYSIS (SCL Float Offset under {clause_map['eot']})
4. FINANCIAL RULING & DELAY DAMAGES (Binding dollar assessment under {clause_map['ld']})
"""
            response = llm.invoke(system_prompt)
            output_text = response.content[0].get("text", "") if isinstance(response.content, list) else response.content

            if not custom_key:
                st.session_state.adjudication_runs += 1

            st.subheader(f"Step 2: Formal Engineer's Determination ({clause_map['determination']})")
            st.markdown(f'<div class="verdict-box">{output_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            pdf_bytes = generate_pdf_report(
                selected_edition, f"FORMAL DETERMINATION ({clause_map['determination']})",
                claim_statement, date_event, date_notice, claimed_days, calc, output_text, clause_map
            )
            st.download_button(
                label=f"📥 Download Formal Determination PDF ({selected_edition})",
                data=pdf_bytes,
                file_name=f"FIDIC_Determination_{'2017' if is_2017 else '1999'}_{date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Execution Error: {e}")

# Path B: Multi-Agent Board
if run_btn_multi:
    if not api_key:
        st.error("API Key missing. Please provide a Gemini API Key.")
        st.stop()

    with st.spinner(f"Deploying Contractor Advocate & Employer Sentry Agents under {selected_edition}..."):
        try:
            results, retrieved_clauses, context_text = get_rag_context(claim_statement, is_2017)
            if not retrieved_clauses:
                st.error("Relevance Gate Audit: No verified contract clauses found. Negotiation aborted.")
                st.stop()

            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

            prompt_contractor = f"""You are the Lead Claims Advocate representing the EPC Contractor under {selected_edition}.
CONTRACT CONTEXT: {context_text}
AUDITED FACTS: Event Inception: {date_event}, Notice Served: {date_notice} ({calc['notice_delta']} days). Claimed EoT: {claimed_days} days. Total project delay: {total_project_delay} days.
DISPUTE NARRATIVE: {claim_statement}

Your Objective: Fight aggressively for full {claimed_days} days EoT under {clause_map['eot']}. Argue why notice was timely under {clause_map['notice']}, reject float ownership, and dispute delay damages under {clause_map['ld']}. Be assertive, structured, and concise (under 200 words)."""
            
            res_contractor = llm.invoke(prompt_contractor)
            contractor_speech = res_contractor.content[0].get("text", "") if isinstance(res_contractor.content, list) else res_contractor.content

            prompt_employer = f"""You are the Legal Sentry defending the Employer under {selected_edition}.
CONTRACT CONTEXT: {context_text}
DETERMINISTIC CPU AUDITED METRICS:
- Notice Delta: {calc['notice_delta']} days (Statutory Limit: 28 days | Time-Bar: {calc['is_time_barred']}) under {clause_map['notice']}
- SCL Float Consumed: {avail_float} days | Concurrent Contractor Delay: {concurrent_delay} days
- Net Defensible EoT: {calc['net_eot']} days under {clause_map['eot']}
- Unexcused Delay: {calc['unexcused_delay']} days
- Contractual Delay Damages Levied: ${calc['liquidated_damages']:,.2f} under {clause_map['ld']}

CONTRACTOR ARGUMENT:
{contractor_speech}

Your Objective: Strictly enforce conditions precedent under {clause_map['notice']}. If notice > 28 days, enforce time-bar unconditionally. Demand full recovery of ${calc['liquidated_damages']:,.2f} delay damages under {clause_map['ld']}. Be decisive and concise (under 200 words)."""

            res_employer = llm.invoke(prompt_employer)
            employer_speech = res_employer.content[0].get("text", "") if isinstance(res_employer.content, list) else res_employer.content

            prompt_accord = f"""You are an accredited Senior Arbitrator / Board Member presiding over this dispute under {selected_edition} {clause_map['board']}.
FACTUAL DETERMINISTIC CPU METRICS: Net EoT: {calc['net_eot']} days ({clause_map['eot']}). Liquidated Damages: ${calc['liquidated_damages']:,.2f} ({clause_map['ld']}). Notice Delta: {calc['notice_delta']} days ({clause_map['notice']}).
CONTRACTOR SUBMISSION: {contractor_speech}
EMPLOYER REBUTTAL: {employer_speech}

Synthesize a binding 'Settlement Accord' under {clause_map['board']}.
Provide:
1. FINAL BINDING EOT AWARD (Days under {clause_map['eot']})
2. FINAL ENFORCEABLE DELAY DAMAGES ($ under {clause_map['ld']})
3. 3-POINT COMMERCIAL SETTLEMENT ACTION PLAN"""

            res_accord = llm.invoke(prompt_accord)
            accord_speech = res_accord.content[0].get("text", "") if isinstance(res_accord.content, list) else res_accord.content

            if not custom_key:
                st.session_state.adjudication_runs += 1

            st.subheader(f"Step 3: Multi-Agent Adversarial Transcript ({selected_edition})")
            tab_view, tab_raw = st.tabs(["🏛️ Adversarial Debate Stream", "📜 Settlement Term Sheet"])

            with tab_view:
                st.markdown(f'<div class="contractor-box"><b>👷 Contractor Advocate Agent:</b><br>{contractor_speech.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="employer-box"><b>🏛️ Employer Sentry Agent:</b><br>{employer_speech.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            with tab_raw:
                st.markdown(f'<div class="accord-box"><b>⚖️ {clause_map["board"]} Mediated Settlement Accord:</b><br>{accord_speech.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            combined_text = f"CONTRACTOR POSITION:\n{contractor_speech}\n\nEMPLOYER REBUTTAL:\n{employer_speech}\n\nBINDING SETTLEMENT ACCORD ({clause_map['board']}):\n{accord_speech}"
            pdf_bytes = generate_pdf_report(
                selected_edition, f"SETTLEMENT ACCORD ({clause_map['board']})",
                claim_statement, date_event, date_notice, claimed_days, calc, combined_text, clause_map
            )
            st.download_button(
                label=f"📥 Download {clause_map['board']} Accord PDF",
                data=pdf_bytes,
                file_name=f"FIDIC_Accord_{'2017' if is_2017 else '1999'}_{date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Negotiation Execution Error: {e}")