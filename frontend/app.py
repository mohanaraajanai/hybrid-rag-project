import os

import httpx
import streamlit as st
from dotenv import load_dotenv


# ==========================================
# 1. Load Environment Variables
# ==========================================

# Find the project root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Load the root .env file.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ==========================================
# 2. Application Configuration
# ==========================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
)


# ==========================================
# 3. Streamlit Page Configuration
# ==========================================

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


# ==========================================
# 4. Application Header
# ==========================================

st.title("🤖 Hybrid RAG Assistant")

st.markdown(
    """
    Welcome to the Hybrid Retrieval-Augmented Generation application.

    This system will combine:

    - FAISS Vector Search
    - Neo4j Knowledge Graph
    - OpenAI LLM
    """
)

st.divider()


# ==========================================
# 5. Backend Health Check
# ==========================================

st.subheader("Backend Connection")

if st.button("Check Backend Status"):

    try:
        response = httpx.get(
            f"{BACKEND_URL}/health",
            timeout=5.0,
        )

        if response.status_code == 200:

            health_data = response.json()

            st.success(
                f"Backend is connected: "
                f"{health_data.get('status')}"
            )

            st.json(health_data)

        else:

            st.error(
                f"Backend returned status code: "
                f"{response.status_code}"
            )

    except httpx.RequestError:

        st.error(
            "Unable to connect to FastAPI. "
            "Make sure the backend server is running."
        )


st.divider()


# ==========================================
# 6. PDF Upload Section
# ==========================================

st.subheader("📄 Upload PDF Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a PDF document for Hybrid RAG processing.",
)

if uploaded_file is not None:

    st.success(
        f"Selected file: {uploaded_file.name}"
    )

    st.write(
        f"File size: "
        f"{uploaded_file.size / 1024:.2f} KB"
    )

    st.info(
        "PDF ingestion API will be connected "
        "in a future step."
    )


st.divider()


# ==========================================
# 7. Ingestion Status Placeholder
# ==========================================

st.subheader("⚙️ Ingestion Status")

st.info(
    "No document has been ingested yet. "
    "The ingestion pipeline will be implemented later."
)


# ==========================================
# 8. Chat Section Placeholder
# ==========================================

st.subheader("💬 Ask Questions")

st.chat_input(
    "Chat will be enabled after successful ingestion.",
    disabled=True,
)

st.caption(
    "Chat is currently disabled until the ingestion "
    "pipeline is completed."
)