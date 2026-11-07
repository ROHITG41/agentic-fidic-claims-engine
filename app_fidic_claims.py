import os
import io
from datetime import date, timedelta
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & SECRETS MANAGEMENT
# ---------------------------------------------------------
st.set_page_config(
    page_title="FIDIC Enterprise Claims & Delay Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Resolve Gemini API Key from environment, Streamlit secrets, or user input
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

# Initialize Session State
if "demo_runs" not in st.session_state:
    st.session_state["demo_runs"] = 0

# Helper to normalize LLM response content (handles string, list of blocks, or dicts)
def extract_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(content)

# ---------------------------------------------------------
# 2. REPORTLAB PDF EXPORT ENGINE WITH AUTHOR ATTRIBUTION
# ---------------------------------------------------------
def clean_pdf_text(text) -> str:
    """Sanitizes strings/lists into Latin-1 compliant characters for ReportLab rendering."""
    # Ensure text is converted to a string even if a list or None is passed
    text = extract_text(text)

    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2022': '*', '•': '*',
        '⚖️': '', '✅': '[COMPLIANT]', '❌': '[TIME-BARRED]', '🔹': '-',
        '🏛️': '', '👷': '', '🤝': '', '📊': '', '📄': '', '⚠️': '[WARN]'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(doc_title, edition, metrics, statement, decision_text, is_accord=False):
    """Generates a formal legal PDF stamped with system author provenance."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        leftMargin=36, 
        rightMargin=36, 
        topMargin=54, 
        bottomMargin=54
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1B365D'),
        spaceAfter=6
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1B365D'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1E293B')
    )
    meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1B365D')
    )
    meta_val = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # 1. Document Title
    story.append(Paragraph(clean_pdf_text(doc_title), title_style))
    story.append(Spacer(1, 4))

    # 2. Author Provenance & Intellectual Property Card
    provenance_data = [
        [
            Paragraph("System Architect & Author:", meta_label),
            Paragraph("<b>ROHIT KUMAR</b>, PMP, CEng, CEA, CEM, FSEr (TUV)", meta_val),
            Paragraph("Direct Inquiries / Cell:", meta_label),
            Paragraph("+91-8826991440 | grohit41@gmail.com", meta_val)
        ],
        [
            Paragraph("LinkedIn Profile:", meta_label),
            Paragraph("linkedin.com/in/rohitg41 (ROHITG41)", meta_val),
            Paragraph("Live Production App:", meta_label),
            Paragraph("rohitkumarg-fidic-engine.streamlit.app", meta_val)
        ],
        [
            Paragraph("Intellectual Property:", meta_label),
            Paragraph("Copyright (c) 2026 Rohit Kumar. All Rights Reserved.", meta_val),
            Paragraph("GitHub Repository:", meta_label),
            Paragraph("github.com/ROHITG41/agentic-fidic-claims-engine", meta_val)
        ]
    ]
    t_prov = Table(provenance_data, colWidths=[110, 160, 100, 170])
    t_prov.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1B365D')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_prov)
    story.append(Spacer(1, 8))

    # 3. Contract Parameter Audit Table
    story.append(Paragraph("1. CONTRACTUAL & DETERMINISTIC AUDIT PARAMETERS", section_heading))
    metrics_data = [
        [
            Paragraph("Governing Contract Edition", meta_label),
            Paragraph(clean_pdf_text(edition), meta_val),
            Paragraph("Notice Timeline Delta", meta_label),
            Paragraph(f"{metrics['notice_delta']} Days (Statutory Limit: 28 Days)", meta_val)
        ],
        [
            Paragraph("Procedural Condition Precedent", meta_label),
            Paragraph("TIME-BARRED" if metrics['is_time_barred'] else "COMPLIANT", meta_val),
            Paragraph("Net Defensible Extension of Time", meta_label),
            Paragraph(f"{metrics['net_eot']} Days Approved", meta_val)
        ],
        [
            Paragraph("Unexcused Delay Assessed", meta_label),
            Paragraph(f"{metrics['unexcused_delay']} Days", meta_val),
            Paragraph("Enforceable Liquidated Damages", meta_label),
            Paragraph(f"${metrics['liquidated_damages']:,.2f} {'(CAPPED)' if metrics['is_capped'] else ''}", meta_val)
        ]
    ]
    t_metrics = Table(metrics_data, colWidths=[130, 140, 130, 140])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFFFF')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 8))

    # 4. Dispute Statement
    story.append(Paragraph("2. FORMAL DISPUTE STATEMENT & OPERATIONAL NARRATIVE", section_heading))
    story.append(Paragraph(clean_pdf_text(statement), body_style))
    story.append(Spacer(1, 8))

    # 5. Formal Legal Ruling or Mediated Accord
    ruling_title = "3. MEDIATED SETTLEMENT ACCORD & TERMS" if is_accord else "3. ENGINEER'S DETERMINATION MEMORANDUM"
    story.append(Paragraph(ruling_title, section_heading))
    
    cleaned_decision = clean_pdf_text(decision_text)
    for para in cleaned_decision.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 4))

    # 6. Engineering Certification & Audit Seal
    story.append(Spacer(1, 10))
    cert_data = [
        [
            Paragraph("<b>CERTIFICATION & AUDIT SEAL:</b> This formal legal determination has been generated by the 3-Bridge Deterministic & Multi-Agent Arbitration Engine.<br/>"
                      "<b>Principal System Architect:</b> Rohit Kumar, PMP, CEng, CEA, CEM, FSEr (TUV)<br/>"
                      "<b>Advisory & Technical Inquiries:</b> grohit41@gmail.com | Cell: +91-8826991440 | LinkedIn: ROHITG41", meta_val)
        ]
    ]
    t_cert = Table(cert_data, colWidths=[540])
    t_cert.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1B365D')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_cert)

    # 7. Running Headers & Footers on Every Page Callback
    def add_running_header_footer(canvas, doc_obj):
        canvas.saveState()
        # Top Running Header
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(colors.HexColor("#1B365D"))
        canvas.drawString(36, 756, "FIDIC ENTERPRISE CLAIMS & DELAY ENGINE -- PRODUCTION AUDIT REPORT")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawRightString(576, 756, "Lead Architect: Rohit Kumar, PMP, CEng")
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.5)
        canvas.line(36, 750, 576, 750)

        # Bottom Running Footer
        canvas.line(36, 45, 576, 45)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(36, 33, "Confidential | Author: Rohit Kumar | Email: grohit41@gmail.com | LinkedIn: ROHITG41")
        canvas.drawRightString(576, 33, f"Page {doc_obj.page} | rohitkumarg-fidic-engine.streamlit.app")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_running_header_footer, onLaterPages=add_running_header_footer)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------
# 3. BRIDGE 2: DETERMINISTIC CPU ARITHMETIC LAYER
# ---------------------------------------------------------
def calculate_delay_and_cost(d_event, d_notice, claimed_days, avail_float, concurrent_delay, total_delay_days, daily_rate, contract_val, cap_pct, notice_clause):
    """Executes statutory date math and SCL delay deductions in pure Python without LLM drift."""
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

# ---------------------------------------------------------
# 4. BRIDGE 1: ROBUST RAG VECTOR ENGINE (HYBRID RETRIEVAL)
# ---------------------------------------------------------
def get_rag_context(statement, edition_name, is_2017, user_api_key):
    """Retrieves contract clauses and filters edition metadata in Python to prevent query planner crashes."""
    persist_dir = "./chroma_fidic_db"
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=user_api_key
    )
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Query Chroma without database-level filter
    search_query = f"{edition_name}: {statement}"
    results = vector_db.similarity_search_with_relevance_scores(search_query, k=8)
    
    # Safe deterministic Python filtering
    target_tag = "2017" if is_2017 else "1999"
    matched = [
        (doc, score) for doc, score in results 
        if doc.metadata.get("edition") == target_tag
    ]
    
    final_results = matched[:3] if matched else results[:3]
    retrieved = [doc for doc, score in final_results if score >= 0.35]
    
    context = "\n\n".join([
        f"--- {doc.metadata.get('clause_number', 'Clause')}: {doc.metadata.get('clause_title', 'Title')} ---\n{doc.page_content}"
        for doc in retrieved
    ])
    return final_results, retrieved, context

# ---------------------------------------------------------
# 5. SIDEBAR: GOVERNING CONTRACT PARAMETERS
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Engine Parameters")
    
    selected_edition = st.selectbox(
        "Select FIDIC Edition:",
        ["FIDIC 2017 Second Edition", "FIDIC 1999 Red Book"],
        index=0
    )
    is_2017 = "2017" in selected_edition
    
    # Dynamic Contract Clause Mapping
    notice_clause = "Sub-Clause 20.2.1" if is_2017 else "Sub-Clause 20.1"
    eot_clause = "Sub-Clause 8.5" if is_2017 else "Sub-Clause 8.4"
    ld_clause = "Sub-Clause 8.8" if is_2017 else "Sub-Clause 8.7"
    engineer_det_clause = "Sub-Clause 3.7" if is_2017 else "Sub-Clause 3.5"
    board_name = "Standing DAAB (Sub-Clause 21.4)" if is_2017 else "Ad-hoc DAB (Sub-Clause 20.4)"

    st.markdown("---")
    st.subheader("Project Financial Parameters")
    contract_sum = st.number_input("Contract Value ($)", min_value=100000.0, value=25000000.0, step=500000.0)
    daily_ld_rate = st.number_input("Daily Delay Damages Rate ($/day)", min_value=100.0, value=15000.0, step=1000.0)
    ld_cap_pct = st.slider("Delay Damages Cap (%)", min_value=1.0, max_value=25.0, value=10.0, step=0.5)

    st.markdown("---")
    st.markdown(f"**Vector Store:** ChromaDB (Multi-Edition)")
    st.markdown(f"**Embeddings:** `gemini-embedding-2`")
    st.markdown(f"**Cognitive Model:** `gemini-3.5-flash`")
    st.markdown(f"**Calculation Layer:** Python CPU (Zero Drift)")

    st.markdown("---")
    user_key_input = st.text_input("Enter Your Own Gemini API Key (Optional)", type="password")
    active_api_key = user_key_input if user_key_input else api_key

# ---------------------------------------------------------
# 6. MAIN WORKSPACE: DISPUTE NARRATIVE & INPUTS
# ---------------------------------------------------------
st.title("⚖️ FIDIC Enterprise Claims & Delay Engine")
st.caption(f"Autonomous 3-Bridge Adjudication Architecture | Governing Edition: **{selected_edition}**")

# Preset Dispute Scenarios
st.markdown("##### Quick Load Real-World Dispute Scenarios:")
col_s1, col_s2, col_s3 = st.columns(3)

default_statement = (
    "Due to encounter with an extensive unforeseen volcanic basalt rock formation at chainage 14+200 "
    "during excavation, works were suspended for 40 calendar days. Contractor submitted notice of claim "
    "within 18 days of encountering the formation, demanding 40 days Extension of Time and site overheads."
)
default_event_date = date.today() - timedelta(days=50)
default_notice_date = default_event_date + timedelta(days=18)
default_claimed_days = 40
default_float = 5
default_concurrent = 7

if col_s1.button("🌋 Scenario 1: Unforeseen Subsurface Rock"):
    default_statement = (
        "Due to encounter with an extensive unforeseen volcanic basalt rock formation at chainage 14+200 "
        "during excavation, works were suspended for 40 calendar days. Contractor submitted notice of claim "
        "within 18 days of encountering the formation, demanding 40 days Extension of Time and site overheads."
    )
    default_claimed_days = 40
    default_float = 5
    default_concurrent = 7

if col_s2.button("⏱️ Scenario 2: Fatal 28-Day Time-Bar"):
    default_statement = (
        "Contractor claims 45 days extension of time resulting from delayed access to the main bridge pier worksite. "
        "Records indicate the employer handed over site on Day 1, but Contractor served formal notice on Day 42."
    )
    default_claimed_days = 45
    default_float = 0
    default_concurrent = 0
    default_notice_date = default_event_date + timedelta(days=42)

if col_s3.button("🌧️ Scenario 3: Adverse Weather & High Concurrency"):
    default_statement = (
        "Contractor claims 30 days EoT due to severe cyclonic rainfall exceeding 100-year return period records. "
        "However, contractor's critical path tower crane was broken down simultaneously for 18 days during the same window."
    )
    default_claimed_days = 30
    default_float = 3
    default_concurrent = 18

claim_statement = st.text_area(
    "Dispute Operational Narrative & Factual Statement:",
    value=default_statement,
    height=110
)

# Deterministic Timeline Inputs
col_d1, col_d2 = st.columns(2)
with col_d1:
    event_date = st.date_input("Date Event/Delay Circumstance Arose:", value=default_event_date)
with col_d2:
    notice_date = st.date_input("Date Notice of Claim Served to Engineer:", value=default_notice_date)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    claimed_days = st.number_input("Claimed EoT (Days):", min_value=1, value=default_claimed_days)
with col_m2:
    avail_float = st.number_input("Available Total Float (Days):", min_value=0, value=default_float)
with col_m3:
    concurrent_delay = st.number_input("Concurrent Contractor Delay (Days):", min_value=0, value=default_concurrent)
with col_m4:
    total_project_delay = st.number_input("Total Project Delay Recorded (Days):", min_value=1, value=claimed_days)

# Execute Bridge 2 Calculation
calc_results = calculate_delay_and_cost(
    event_date, notice_date, claimed_days, avail_float, concurrent_delay,
    total_project_delay, daily_ld_rate, contract_sum, ld_cap_pct, notice_clause
)

# Display Real-Time Deterministic Scorecards
st.markdown("---")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        label=f"Notice Timeline ({notice_clause})",
        value=f"{calc_results['notice_delta']} Days",
        delta="Procedurally Valid" if not calc_results['is_time_barred'] else "Time-Barred (>28d)",
        delta_color="normal" if not calc_results['is_time_barred'] else "inverse"
    )
with kpi2:
    st.metric(
        label=f"Defensible EoT ({eot_clause})",
        value=f"{calc_results['net_eot']} Days",
        delta=f"Float: -{avail_float}d | Concurr: -{concurrent_delay}d"
    )
with kpi3:
    st.metric(
        label="Unexcused Contractor Delay",
        value=f"{calc_results['unexcused_delay']} Days",
        delta="Assessable for Delay Damages",
        delta_color="inverse" if calc_results['unexcused_delay'] > 0 else "off"
    )
with kpi4:
    st.metric(
        label=f"Liquidated Damages ({ld_clause})",
        value=f"${calc_results['liquidated_damages']:,.2f}",
        delta="CAPPED AT MAX LIMIT" if calc_results['is_capped'] else "Daily Accrual",
        delta_color="inverse" if calc_results['liquidated_damages'] > 0 else "normal"
    )

if calc_results['is_time_barred']:
    st.error(f"🚨 **FATAL CONDITION PRECEDENT VIOLATION:** Notice was served {calc_results['notice_delta']} days after event awareness, exceeding the mandatory 28-day window under {notice_clause}. Claim is statutorily time-barred.")
else:
    st.success(f"✅ **PROCEDURAL COMPLIANCE VERIFIED:** Notice served within {calc_results['notice_delta']} days. Entitlement evaluated under {eot_clause} after float and concurrency offsets.")

st.markdown("---")

# ---------------------------------------------------------
# 7. PATH A: BRIDGE 2 ENGINEER'S FORMAL DETERMINATION
# ---------------------------------------------------------
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    run_determination_clicked = st.button(
        f"⚖️ Run Determination ({engineer_det_clause} & Zero-Drift Math)",
        use_container_width=True,
        type="primary"
    )

with btn_col2:
    run_board_clicked = st.button(
        f"🤝 Run Bridge 3: Multi-Agent Negotiation ({board_name})",
        use_container_width=True
    )

if run_determination_clicked:
    if not active_api_key:
        st.error("Please provide a valid Gemini API Key in the sidebar or environment.")
    else:
        with st.spinner("Retrieving verified contract provisions & formulating determination..."):
            try:
                # 1. RAG Retrieval
                raw_results, retrieved_docs, rag_context = get_rag_context(claim_statement, selected_edition, is_2017, active_api_key)
                
                # 2. LLM Cognitive Synthesis
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash",
                    temperature=0.0,
                    google_api_key=active_api_key
                )
                
                system_prompt = (
                    f"You are the Accredited Engineer presiding over a contract governed by the {selected_edition}.\n"
                    f"You are bound strictly to act neutrally and independently under {engineer_det_clause}.\n"
                    f"You are given non-negotiable mathematical facts calculated by the Python Deterministic Layer:\n"
                    f"- Notice Delta: {calc_results['notice_delta']} days (Limit: 28 days).\n"
                    f"- Time-Bar Status: {'FATALLY REJECTED' if calc_results['is_time_barred'] else 'PROCEDURALLY COMPLIANT'}.\n"
                    f"- Net Defensible Extension of Time: {calc_results['net_eot']} calendar days.\n"
                    f"- Unexcused Delay: {calc_results['unexcused_delay']} calendar days.\n"
                    f"- Assessable Liquidated Damages: ${calc_results['liquidated_damages']:,.2f} ({'Capped' if calc_results['is_capped'] else 'Standard'}).\n\n"
                    f"Retrieved Contract Context:\n{rag_context}\n\n"
                    f"Draft a formal, authoritative Engineer's Determination Letter addressing the Contractor and Employer. "
                    f"Cite specific sub-clauses, state the procedural ruling, detail the delay analysis, and conclude with binding instructions."
                )
                
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Claim Narrative:\n{claim_statement}")
                ])
                
                # Use extract_text to convert any list/content blocks into clean string
                st.session_state["determination_letter"] = extract_text(response.content)
                st.session_state["determination_rag"] = rag_context
                st.session_state["determination_edition"] = selected_edition

            except Exception as e:
                st.error(f"Execution Error: {e}")

if "determination_letter" in st.session_state:
    st.markdown(f"### 📋 Engineer's Formal Determination ({st.session_state['determination_edition']})")
    st.markdown(st.session_state["determination_letter"])
    
    with st.expander("🔍 View Retrieved Contract Ground-Truth Clauses"):
        st.markdown(st.session_state["determination_rag"])
    
    # Generate Downloadable PDF with Author Provenance
    pdf_bytes = generate_pdf_report(
        doc_title=f"ENGINEER'S FORMAL DETERMINATION ({st.session_state['determination_edition']})",
        edition=st.session_state["determination_edition"],
        metrics=calc_results,
        statement=claim_statement,
        decision_text=st.session_state["determination_letter"],
        is_accord=False
    )
    
    st.download_button(
        label="📥 Download Determination Letter (Branded Audit PDF)",
        data=pdf_bytes,
        file_name="FIDIC_Engineers_Determination_Audit.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# ---------------------------------------------------------
# 8. PATH B: BRIDGE 3 MULTI-AGENT ADVERSARIAL ACCORD
# ---------------------------------------------------------
if run_board_clicked:
    if not active_api_key:
        st.error("Please provide a valid Gemini API Key in the sidebar or environment.")
    else:
        with st.spinner("Convening Standing Dispute Board & conducting adversarial arbitration..."):
            try:
                # 1. RAG Retrieval
                raw_results, retrieved_docs, rag_context = get_rag_context(claim_statement, selected_edition, is_2017, active_api_key)
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash",
                    temperature=0.2,
                    google_api_key=active_api_key
                )

                # Agent A: Contractor Advocate
                contractor_prompt = (
                    f"You are the Contractor's Senior Claims Advocate under {selected_edition}.\n"
                    f"Advocate aggressively for maximum Extension of Time under {eot_clause} and challenge all float offsets.\n"
                    f"Argue that notice was served as soon as practicable and that critical path was fully impacted.\n"
                    f"Facts: Claimed {claimed_days}d, Float: {avail_float}d, Concurrent: {concurrent_delay}d.\n"
                    f"Ground Truth:\n{rag_context}\n\nPresent the Contractor's Formal Submittal."
                )
                res_contractor = extract_text(llm.invoke([
                    SystemMessage(content=contractor_prompt),
                    HumanMessage(content=claim_statement)
                ]).content)

                # Agent B: Employer Legal Sentry
                employer_prompt = (
                    f"You are the Employer's Commercial & Legal Sentry under {selected_edition}.\n"
                    f"Enforce the strict condition precedent of {notice_clause}. If notice exceeded 28 days, demand complete forfeiture.\n"
                    f"Demand assessment of Delay Damages under {ld_clause} for {calc_results['unexcused_delay']} days totaling ${calc_results['liquidated_damages']:,.2f}.\n"
                    f"Ground Truth:\n{rag_context}\n\nPresent the Employer's Statement of Defense."
                )
                res_employer = extract_text(llm.invoke([
                    SystemMessage(content=employer_prompt),
                    HumanMessage(content=f"Contractor's Submission:\n{res_contractor}\n\nFacts: {calc_results['rejection_reason']}")
                ]).content)

                # Agent C: Presiding Arbitrator Accord
                arbitrator_prompt = (
                    f"You are the Presiding Neutral Member of the {board_name} under {selected_edition}.\n"
                    f"You have listened to the Contractor and Employer debate. You are strictly bound by the mathematical audit:\n"
                    f"- Net Defensible EoT: {calc_results['net_eot']} days.\n"
                    f"- Liquidated Damages: ${calc_results['liquidated_damages']:,.2f}.\n"
                    f"- Procedural Status: {calc_results['rejection_reason']}.\n"
                    f"Draft a formal, binding 3-point Mediated Settlement Accord balancing commercial progress with contractual compliance."
                )
                res_arbitrator = extract_text(llm.invoke([
                    SystemMessage(content=arbitrator_prompt),
                    HumanMessage(content=f"Debate Summary:\nContractor: {res_contractor}\n\nEmployer: {res_employer}")
                ]).content)

                st.session_state["board_contractor"] = res_contractor
                st.session_state["board_employer"] = res_employer
                st.session_state["board_accord"] = res_arbitrator
                st.session_state["board_edition"] = selected_edition

            except Exception as e:
                st.error(f"Arbitration Error: {e}")

if "board_accord" in st.session_state:
    st.markdown(f"### 🏛️ {board_name} -- Adversarial Proceedings")
    tab_cont, tab_emp, tab_accord = st.tabs([
        "👷 Contractor Advocate Submittal",
        "🏛️ Employer Legal Sentry Defense",
        "📜 Enforceable Settlement Accord"
    ])
    
    with tab_cont:
        st.markdown(st.session_state["board_contractor"])
    with tab_emp:
        st.markdown(st.session_state["board_employer"])
    with tab_accord:
        st.markdown(st.session_state["board_accord"])
    
    # Generate Downloadable Accord PDF
    pdf_accord_bytes = generate_pdf_report(
        doc_title=f"MEDIATED DISPUTE SETTLEMENT ACCORD ({st.session_state['board_edition']})",
        edition=st.session_state["board_edition"],
        metrics=calc_results,
        statement=claim_statement,
        decision_text=st.session_state["board_accord"],
        is_accord=True
    )
    
    st.download_button(
        label="📥 Download Mediated Settlement Accord (Branded Audit PDF)",
        data=pdf_accord_bytes,
        file_name="FIDIC_Dispute_Board_Settlement_Accord.pdf",
        mime="application/pdf",
        use_container_width=True
    )