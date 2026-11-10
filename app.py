import os
from pathlib import Path

import streamlit as st
import numpy as np
import requests

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Pneumonia AI Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pneumonia_cnn_model.h5"

GDRIVE_FILE_ID = "1nP1sWihrPi8v9pAdFMpQDBkUTycm_17t"

PAGES = [
    "🏠  Overview",
    "🩻  X-ray Detection",
    "🤖  AI Knowledge Assistant",
    "ℹ️  About Project",
]

SUGGESTED_QUESTIONS = [
    "What is pneumonia?",
    "What are the symptoms?",
    "What causes pneumonia?",
    "How can pneumonia be prevented?",
    "What is the difference between normal and pneumonia-affected lungs?",
]


# ==========================================
# 2. CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0B2545;
        --navy-2: #13315C;
        --teal: #0E9AA7;
        --teal-dark: #0A7C86;
        --cyan: #7FDBFF;
        --cyan-soft: #E3F6FC;
        --bg: #F3F8FB;
        --card: #FFFFFF;
        --text: #1B2B3A;
        --muted: #64778A;
        --border: #E2EBF1;
        --green: #12A150;
        --green-bg: #E8F7EE;
        --red: #D9480F;
        --red-bg: #FFF0E8;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B2545 0%, #13315C 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #E6F1F7;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 0 14px 0;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0E9AA7, #7FDBFF);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 6px 16px rgba(14, 154, 167, 0.45);
    }

    .brand-name {
        font-weight: 700;
        font-size: 17px;
        line-height: 1.15;
        color: #FFFFFF;
    }

    .brand-sub {
        font-size: 11.5px;
        color: #9FC3D6;
        letter-spacing: 0.4px;
    }

    .side-heading {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #7FB4CB !important;
        margin: 6px 0 8px 0;
        font-weight: 600;
    }

    .side-info {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 12.5px;
        line-height: 1.7;
    }

    .side-info b {
        color: #7FDBFF;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 10px 14px;
        transition: all 0.18s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(127, 219, 255, 0.12);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(14, 154, 167, 0.55), rgba(127, 219, 255, 0.18));
        border-color: rgba(127, 219, 255, 0.45);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] p {
        font-size: 14.5px;
        font-weight: 500;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(120deg, #0B2545 0%, #13315C 55%, #0E9AA7 130%);
        border-radius: 22px;
        padding: 38px 42px;
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
        box-shadow: 0 14px 34px rgba(11, 37, 69, 0.22);
        margin-bottom: 26px;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -60px;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(127, 219, 255, 0.35), transparent 70%);
    }

    .hero-pill {
        display: inline-block;
        background: rgba(127, 219, 255, 0.16);
        border: 1px solid rgba(127, 219, 255, 0.4);
        color: #BDEEFF;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
    }

    .hero h1 {
        font-size: 40px;
        font-weight: 800;
        margin: 0 0 10px 0;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }

    .hero p {
        font-size: 16.5px;
        color: #D3E7F2;
        max-width: 640px;
        line-height: 1.6;
        margin: 0;
    }

    /* ---------- Page headers ---------- */
    .page-title {
        font-size: 28px;
        font-weight: 800;
        color: var(--navy);
        margin-bottom: 2px;
        letter-spacing: -0.3px;
    }

    .page-sub {
        color: var(--muted);
        font-size: 15px;
        margin-bottom: 22px;
    }

    /* ---------- Cards ---------- */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 26px;
        box-shadow: 0 4px 18px rgba(11, 37, 69, 0.06);
        height: 100%;
    }

    .card-lift {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card-lift:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(11, 37, 69, 0.12);
    }

    .card-icon {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: var(--cyan-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        margin-bottom: 16px;
    }

    .card h3 {
        font-size: 19px;
        font-weight: 700;
        color: var(--navy);
        margin: 0 0 8px 0;
    }

    .card p {
        color: var(--muted);
        font-size: 14.5px;
        line-height: 1.65;
        margin: 0 0 12px 0;
    }

    .card ul {
        margin: 0;
        padding-left: 18px;
        color: var(--text);
        font-size: 14px;
        line-height: 1.9;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--navy);
        margin: 28px 0 14px 0;
    }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-block;
        background: var(--cyan-soft);
        color: var(--teal-dark);
        border: 1px solid #BFE6F1;
        padding: 7px 15px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin: 0 8px 8px 0;
    }

    /* ---------- Disclaimer ---------- */
    .disclaimer {
        background: #FFF8EB;
        border: 1px solid #FFE1A8;
        border-left: 5px solid #F59E0B;
        border-radius: 12px;
        padding: 15px 18px;
        font-size: 13.8px;
        line-height: 1.65;
        color: #6B4B0A;
        margin-top: 24px;
    }

    /* ---------- Result cards ---------- */
    .result-card {
        border-radius: 18px;
        padding: 26px;
        border: 1px solid var(--border);
        background: var(--card);
        box-shadow: 0 4px 18px rgba(11, 37, 69, 0.06);
    }

    .result-normal {
        border-top: 6px solid var(--green);
        background: linear-gradient(180deg, var(--green-bg), #FFFFFF 55%);
    }

    .result-pneu {
        border-top: 6px solid var(--red);
        background: linear-gradient(180deg, var(--red-bg), #FFFFFF 55%);
    }

    .result-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--muted);
        font-weight: 600;
    }

    .result-status {
        font-size: 36px;
        font-weight: 800;
        margin: 4px 0 14px 0;
        letter-spacing: -0.5px;
    }

    .status-normal { color: var(--green); }
    .status-pneu { color: var(--red); }

    .score-row {
        display: flex;
        justify-content: space-between;
        font-size: 13.5px;
        color: var(--text);
        font-weight: 600;
        margin-bottom: 6px;
    }

    .bar-bg {
        background: #E7EEF3;
        height: 10px;
        border-radius: 999px;
        overflow: hidden;
    }

    .bar-fill-normal {
        height: 100%;
        background: linear-gradient(90deg, #12A150, #5BD68B);
        border-radius: 999px;
    }

    .bar-fill-pneu {
        height: 100%;
        background: linear-gradient(90deg, #D9480F, #FF9F68);
        border-radius: 999px;
    }

    .score-note {
        font-size: 12.5px;
        color: var(--muted);
        line-height: 1.6;
        margin-top: 14px;
    }

    /* ---------- Sources ---------- */
    .source-card {
        background: #F7FBFD;
        border: 1px solid var(--border);
        border-left: 4px solid var(--teal);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13.5px;
        line-height: 1.7;
    }

    .sim-pill {
        display: inline-block;
        background: var(--cyan-soft);
        color: var(--teal-dark);
        border-radius: 999px;
        padding: 1px 10px;
        font-weight: 600;
        font-size: 12.5px;
    }

    /* ---------- Streamlit widget polish ---------- */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
        font-weight: 600;
        transition: all 0.18s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0E9AA7, #13315C);
        color: #FFFFFF;
        border: none;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 6px 16px rgba(14, 154, 167, 0.3);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(14, 154, 167, 0.4);
        color: #FFFFFF;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: var(--teal);
        color: var(--teal-dark);
        background: var(--cyan-soft);
    }

    [data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 2px dashed #9ED8E6;
        border-radius: 16px;
        padding: 22px;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--teal);
        background: #F5FCFE;
    }

    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(11, 37, 69, 0.04);
    }

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 13px;
        margin-top: 36px;
        padding-top: 18px;
        border-top: 1px solid var(--border);
        line-height: 1.8;
    }

    @media (max-width: 768px) {
        .hero { padding: 26px 22px; }
        .hero h1 { font-size: 28px; }
        .result-status { font-size: 28px; }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD CNN MODEL
# ==========================================

@st.cache_resource
def get_cnn_model():

    if not MODEL_PATH.exists():

        import gdown

        url = (
            f"https://drive.google.com/uc?id="
            f"{GDRIVE_FILE_ID}"
        )

        gdown.download(
            url,
            str(MODEL_PATH),
            quiet=False
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "CNN model could not be downloaded."
        )

    return load_model(
        str(MODEL_PATH),
        compile=False
    )


# ==========================================
# 4. LOAD RAG MODULE
# ==========================================

@st.cache_resource
def get_rag_module():

    import rag

    return rag


# ==========================================
# 5. HELPERS
# ==========================================

DISCLAIMER_HTML = """
<div class="disclaimer">
<b>⚠️ Medical disclaimer:</b> The CNN is an academic demonstration and is
<b>not</b> a clinically validated diagnostic tool. The chatbot provides
educational information only. Neither output replaces professional medical
evaluation. Always consult a qualified healthcare professional for any
medical decision.
</div>
"""


def navigate_to(page):
    """Change the sidebar page before Streamlit recreates its widgets."""
    st.session_state.nav = page


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True
    )


def render_sources(sources):
    """Render retrieved sources inside an expander."""

    with st.expander(f"📚 Retrieved sources ({len(sources)})"):

        for source in sources:

            st.markdown(
                f"""
                <div class="source-card">
                    <b>📄 {source['source']}</b><br>
                    Chunk: <b>{source['chunk']}</b> &nbsp;·&nbsp;
                    Similarity:
                    <span class="sim-pill">{source['similarity']:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


def answer_question(question):
    """Run the existing RAG pipeline and store the result."""

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🫁"):

        try:

            with st.spinner(
                "Searching knowledge base and "
                "generating answer..."
            ):

                rag = get_rag_module()

                answer, sources = rag.ask_question(
                    question,
                    top_k=3
                )

            st.markdown(answer)

            render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except requests.exceptions.ConnectionError:

            error_message = (
                "Could not connect to Ollama. "
                "Please make sure Ollama is running "
                "at http://localhost:11434."
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })

        except Exception as error:

            error_message = (
                f"Something went wrong: {error}"
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })


# ==========================================
# 6. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "xray_result" not in st.session_state:
    st.session_state.xray_result = None

if "nav" not in st.session_state:
    st.session_state.nav = PAGES[0]


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🫁</div>
            <div>
                <div class="brand-name">Pneumonia AI</div>
                <div class="brand-sub">CNN + RAG ASSISTANT</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-heading">Navigation</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        PAGES,
        key="nav",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="side-heading">Model information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-info">
            <b>Detector:</b> Custom CNN<br>
            <b>Input:</b> 150 × 150 RGB<br>
            <b>Classes:</b> NORMAL / PNEUMONIA<br>
            <b>Threshold:</b> 0.5<br>
            <b>Assistant:</b> Llama 3.2 (Ollama)
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-heading" style="margin-top:16px;">'
        'Technology stack</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-info">
            TensorFlow / Keras<br>
            FAISS<br>
            Sentence Transformers<br>
            Ollama<br>
            Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-size:11.5px; color:#8DB6CA; '
        'margin-top:18px; line-height:1.6;">'
        'Academic demonstration only.<br>Not for clinical use.'
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================
# 8. OVERVIEW PAGE
# ==========================================

def render_overview():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-pill">🩺 HEALTHCARE AI DASHBOARD</div>
            <h1>Pneumonia AI Assistant</h1>
            <p>
                A deep-learning chest X-ray classifier paired with a
                retrieval-augmented medical knowledge assistant — two AI
                systems in one clean, focused workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown(
            """
            <div class="card card-lift">
                <div class="card-icon">🩻</div>
                <h3>CNN Chest X-ray Detection</h3>
                <p>
                    Upload a chest X-ray and let the trained
                    convolutional neural network classify it.
                </p>
                <ul>
                    <li>NORMAL / PNEUMONIA classification</li>
                    <li>150 × 150 preprocessing, scaled to 0–1</li>
                    <li>Instant model output score</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.button(
            "Open X-ray Detection →",
            key="go_detect",
            use_container_width=True,
            on_click=navigate_to,
            args=(PAGES[1],)
        )

    with col2:

        st.markdown(
            """
            <div class="card card-lift">
                <div class="card-icon">🤖</div>
                <h3>RAG Medical Knowledge Assistant</h3>
                <p>
                    Ask educational questions and get answers grounded
                    in the project's knowledge base.
                </p>
                <ul>
                    <li>Top-3 semantic retrieval with FAISS</li>
                    <li>Local answer generation with Llama 3.2</li>
                    <li>Transparent source documents</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.button(
            "Open Knowledge Assistant →",
            key="go_chat",
            use_container_width=True,
            on_click=navigate_to,
            args=(PAGES[2],)
        )

    st.markdown(
        '<div class="section-title">Built with</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <span class="badge">🧠 TensorFlow</span>
        <span class="badge">🔎 FAISS</span>
        <span class="badge">🔤 Sentence Transformers</span>
        <span class="badge">🦙 Ollama</span>
        <span class="badge">🎈 Streamlit</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


# ==========================================
# 9. X-RAY DETECTION PAGE
# ==========================================

def render_detection():

    page_header(
        "🩻 Chest X-ray Detection",
        "Upload a chest X-ray image to get a prediction "
        "from the trained CNN model."
    )

    uploaded_file = st.file_uploader(
        "Drag and drop a chest X-ray image, or browse",
        type=["jpg", "jpeg", "png"],
        key="xray_uploader"
    )

    if uploaded_file is None:

        st.session_state.xray_result = None

        st.info("Upload a chest X-ray image to begin.")

        st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

        return

    try:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as error:

        st.error(f"Could not open image: {error}")

        return

    file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown(
            '<div class="section-title" style="margin-top:0;">'
            'Uploaded X-ray</div>',
            unsafe_allow_html=True
        )

        st.image(
            original_image,
            use_container_width=True
        )

        analyze = st.button(
            "🔬 Analyze X-ray",
            type="primary",
            use_container_width=True,
            key="analyze_btn"
        )

    if analyze:

        try:

            with st.spinner(
                "Loading CNN model and analyzing image..."
            ):

                model = get_cnn_model()

                resized_image = original_image.resize(
                    (150, 150)
                )

                img_array = image.img_to_array(
                    resized_image
                ) / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = float(
                    model.predict(
                        img_array,
                        verbose=0
                    )[0][0]
                )

            label = (
                "PNEUMONIA"
                if prediction >= 0.5
                else "NORMAL"
            )

            model_score = (
                prediction * 100
                if label == "PNEUMONIA"
                else (1 - prediction) * 100
            )

            st.session_state.xray_result = {
                "file_key": file_key,
                "label": label,
                "score": model_score,
                "raw": prediction
            }

        except Exception as error:

            st.session_state.xray_result = None

            with col2:
                st.error(
                    f"Image prediction failed: {error}"
                )

    result = st.session_state.xray_result

    with col2:

        st.markdown(
            '<div class="section-title" style="margin-top:0;">'
            'CNN Prediction</div>',
            unsafe_allow_html=True
        )

        if result and result["file_key"] == file_key:

            is_pneu = result["label"] == "PNEUMONIA"

            card_cls = "result-pneu" if is_pneu else "result-normal"
            status_cls = "status-pneu" if is_pneu else "status-normal"
            bar_cls = "bar-fill-pneu" if is_pneu else "bar-fill-normal"
            icon = "⚠️" if is_pneu else "✅"

            st.markdown(
                f"""
                <div class="result-card {card_cls}">
                    <div class="result-label">Model classification</div>
                    <div class="result-status {status_cls}">
                        {icon} {result['label']}
                    </div>
                    <div class="score-row">
                        <span>Model output score</span>
                        <span>{result['score']:.2f}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="{bar_cls}"
                             style="width:{result['score']:.2f}%;"></div>
                    </div>
                    <div class="score-note">
                        Raw sigmoid output: <b>{result['raw']:.4f}</b>
                        (threshold 0.5).<br>
                        This score reflects the model's output only. It is
                        <b>not</b> a clinically validated probability or
                        diagnostic confidence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="card" style="text-align:center;">
                    <div style="font-size:42px;">🩺</div>
                    <p style="margin-top:8px;">
                        Press <b>Analyze X-ray</b> to run the CNN on
                        this image.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


# ==========================================
# 10. AI KNOWLEDGE ASSISTANT PAGE
# ==========================================

def render_chatbot():

    page_header(
        "🤖 AI Knowledge Assistant",
        "Ask educational questions about pneumonia. Answers are "
        "generated from the project's knowledge base using "
        "FAISS retrieval and Llama 3.2."
    )

    top_col1, top_col2 = st.columns([5, 1])

    with top_col2:

        if st.button(
            "🗑️ Clear",
            key="clear_chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

    with top_col1:

        st.caption(
            "Educational information only — not medical advice."
        )

    # Suggested questions (shown while chat is empty)

    if not st.session_state.messages:

        st.markdown(
            '<div class="section-title" style="margin-top:6px;">'
            '💡 Suggested questions</div>',
            unsafe_allow_html=True
        )

        for i, suggestion in enumerate(SUGGESTED_QUESTIONS):

            if st.button(
                suggestion,
                key=f"suggest_{i}",
                use_container_width=True
            ):
                st.session_state.pending_question = suggestion
                st.rerun()

    # Conversation history

    for message in st.session_state.messages:

        avatar = "🧑" if message["role"] == "user" else "🫁"

        with st.chat_message(message["role"], avatar=avatar):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                render_sources(message["sources"])

    # Pending suggested question or typed question

    typed_question = st.chat_input(
        "Ask a question about pneumonia..."
    )

    question = st.session_state.pending_question or typed_question

    st.session_state.pending_question = None

    if question:
        answer_question(question)


# ==========================================
# 11. ABOUT PAGE
# ==========================================

def render_about():

    page_header(
        "ℹ️ About Project",
        "How the Pneumonia AI Assistant is put together."
    )

    st.markdown(
        """
        <div class="card" style="margin-bottom:20px;">
            <h3>Project description</h3>
            <p>
                Pneumonia AI Assistant combines a custom convolutional
                neural network for chest X-ray classification with a
                retrieval-augmented generation (RAG) assistant that
                answers educational questions about pneumonia. It was
                built as an academic project to demonstrate how
                computer vision and language models can work side by
                side in a healthcare-themed application.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown(
            """
            <div class="card">
                <div class="card-icon">🧠</div>
                <h3>CNN architecture</h3>
                <ul>
                    <li>Framework: TensorFlow / Keras</li>
                    <li>Custom convolutional neural network</li>
                    <li>Input: chest X-ray resized to 150 × 150, divided by 255</li>
                    <li>Output: single sigmoid score</li>
                    <li>Decision: PNEUMONIA if score ≥ 0.5, else NORMAL</li>
                    <li>Weights: <code>pneumonia_cnn_model.h5</code>
                        (auto-downloaded from Google Drive if missing)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="card-icon">🔎</div>
                <h3>RAG architecture</h3>
                <ul>
                    <li>Knowledge base: text documents in <code>knowledge_base/</code></li>
                    <li>Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)</li>
                    <li>Vector search: FAISS, top 3 chunks</li>
                    <li>LLM: Llama 3.2 via local Ollama API</li>
                    <li>Answers shown with document, chunk and similarity</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Technology stack</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <span class="badge">🐍 Python</span>
        <span class="badge">🧠 TensorFlow / Keras</span>
        <span class="badge">🔎 FAISS</span>
        <span class="badge">🔤 Sentence Transformers</span>
        <span class="badge">🦙 Ollama · Llama 3.2</span>
        <span class="badge">🎈 Streamlit</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Developers</div>',
        unsafe_allow_html=True
    )

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown(
            """
            <div class="card card-lift">
                <h3>👨‍💻 Manmeet Singh</h3>
                <p>Developer<br>Punjabi University, Patiala</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d2:
        st.markdown(
            """
            <div class="card card-lift">
                <h3>👨‍💻 Gurkirat Singh</h3>
                <p>Developer<br>Punjabi University, Patiala</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


# ==========================================
# 12. ROUTER
# ==========================================

if page == PAGES[0]:
    render_overview()
elif page == PAGES[1]:
    render_detection()
elif page == PAGES[2]:
    render_chatbot()
else:
    render_about()


# ==========================================
# 13. FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer-note">
        Pneumonia AI Assistant · CNN + RAG + Llama 3.2<br>
        Developed by Manmeet Singh &amp; Gurkirat Singh ·
        Punjabi University, Patiala
    </div>
    """,
    unsafe_allow_html=True
)
