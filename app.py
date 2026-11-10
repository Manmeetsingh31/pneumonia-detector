
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
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pneumonia_cnn_model.h5"

GDRIVE_FILE_ID = "1nP1sWihrPi8v9pAdFMpQDBkUTycm_17t"


# ==========================================
# 2. CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        color: #2E86C1;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .disclaimer {
        background-color: rgba(255, 165, 0, 0.10);
        padding: 14px;
        border-radius: 10px;
        border-left: 4px solid orange;
        font-size: 14px;
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
# 5. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '🫁 Pneumonia AI Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-based Chest X-ray Analysis + '
    'Retrieval-Augmented Generation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 6. APPLICATION TABS
# ==========================================

home_tab, detection_tab, chatbot_tab = st.tabs([
    "🏠 Home",
    "🩻 X-ray Detection",
    "🤖 AI Knowledge Assistant"
])


# ==========================================
# 7. HOME TAB
# ==========================================

with home_tab:

    st.header("Welcome to Pneumonia AI Assistant")

    st.write(
        """
        This application combines two AI technologies
        in one platform:

        **1. CNN-based Pneumonia Detection**

        Upload a chest X-ray image and use a trained
        Convolutional Neural Network to classify it
        as NORMAL or PNEUMONIA.

        **2. RAG-powered Knowledge Assistant**

        Ask questions about pneumonia. The system
        retrieves relevant information from its
        knowledge base using FAISS and generates
        answers using Llama 3.2.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🩻 CNN Detection")

        st.write(
            "Analyze chest X-ray images using "
            "your trained deep learning model."
        )

        st.info(
            "Model: TensorFlow / Keras CNN"
        )

    with col2:

        st.subheader("🤖 RAG Assistant")

        st.write(
            "Ask educational questions and receive "
            "answers grounded in your knowledge base."
        )

        st.info(
            "Technology: Sentence Transformers + "
            "FAISS + Ollama"
        )

    st.divider()

    st.markdown("""
    <div class="disclaimer">
    <b>Medical disclaimer:</b> This application is
    an educational and academic demonstration.
    Its predictions and chatbot responses are not
    medical diagnoses. Always consult qualified
    healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 8. CNN DETECTION TAB
# ==========================================

with detection_tab:

    st.header("Chest X-ray Pneumonia Detection")

    st.write(
        "Upload a chest X-ray image to obtain "
        "a prediction from the trained CNN model."
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        key="xray_uploader"
    )

    if uploaded_file is not None:

        try:

            original_image = Image.open(
                uploaded_file
            ).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Uploaded X-ray")

                st.image(
                    original_image,
                    use_container_width=True
                )

            with col2:

                st.subheader("CNN Prediction")

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

                if label == "PNEUMONIA":

                    st.error(
                        "Prediction: PNEUMONIA"
                    )

                else:

                    st.success(
                        "Prediction: NORMAL"
                    )

                st.metric(
                    "Model score",
                    f"{model_score:.2f}%"
                )

                st.caption(
                    "This score reflects the model's "
                    "output, not a clinically validated "
                    "probability or diagnostic confidence."
                )

        except Exception as error:

            st.error(
                f"Image prediction failed: {error}"
            )

    else:

        st.info(
            "Upload a chest X-ray image to begin."
        )


# ==========================================
# 9. RAG CHATBOT TAB
# ==========================================

with chatbot_tab:

    st.header("Pneumonia Knowledge Assistant")

    st.write(
        """
        Ask questions about pneumonia, its symptoms,
        causes, prevention and related topics.

        Answers are generated using information
        retrieved from the project's knowledge base.
        """
    )

    st.caption(
        "Powered by FAISS retrieval and Llama 3.2."
    )

    # Initialize chat history

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # Clear chat button

    if st.button(
        "🗑️ Clear conversation",
        key="clear_chat"
    ):

        st.session_state.messages = []

        st.rerun()

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📚 Retrieved sources"
                ):

                    for source in message["sources"]:

                        st.markdown(
                            f"""
                            **File:** {source['source']}

                            **Chunk:** {source['chunk']}

                            **Similarity:** {
                                source['similarity']:.4f
                            }
                            """
                        )

    # User question input

    user_question = st.chat_input(
        "Ask a question about pneumonia..."
    )

    if user_question:

        # Display user message

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):

            st.markdown(user_question)

        # Generate answer

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching knowledge base and "
                    "generating answer..."
                ):

                    rag = get_rag_module()

                    answer, sources = rag.ask_question(
                        user_question,
                        top_k=3
                    )

                st.markdown(answer)

                # Display retrieved sources

                with st.expander(
                    "📚 Retrieved sources"
                ):

                    for source in sources:

                        st.markdown(
                            f"""
                            **File:** {source['source']}

                            **Chunk:** {source['chunk']}

                            **Similarity:** {
                                source['similarity']:.4f
                            }

                            ---
                            """
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except requests.exceptions.ConnectionError:

                error_message = (
                    "Could not connect to Ollama. "
                    "Please make sure Ollama is running."
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
# 10. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        Pneumonia AI Assistant |
        CNN + RAG + Llama 3.2

        <br><br>

        Developed by Amanpreet Singh and
        Gurkirat Singh

        <br>
        Punjabi University, Patiala
    </div>
    """,
    unsafe_allow_html=True
)
