import os
from datetime import datetime

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

HEALTH_ENDPOINT = f"{BACKEND_URL}/"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/api/upload/pdf"
QUERY_ENDPOINT = f"{BACKEND_URL}/api/query"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777777;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .status-box {
            padding: 0.9rem;
            border-radius: 0.6rem;
            background-color: #f0f2f6;
            margin-bottom: 1rem;
        }

        .document-box {
            padding: 0.8rem;
            border-radius: 0.6rem;
            background-color: #f0f2f6;
            margin-top: 0.5rem;
        }

        .source-box {
            padding: 0.8rem;
            border-radius: 0.5rem;
            border: 1px solid #dddddd;
            margin-bottom: 0.6rem;
        }

        .small-text {
            font-size: 0.85rem;
            color: #777777;
        }

        .success-text {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "current_document" not in st.session_state:
    st.session_state.current_document = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_seconds(value):
    """
    Format timing values safely.
    """

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return str(value)


def get_backend_error_message(response):
    """
    Convert an HTTP/API error into a user-friendly message.

    Special handling is included for OpenAI/API rate-limit
    errors so raw provider messages are not exposed in the UI.
    """

    raw_text = ""

    try:
        raw_text = response.text or ""
    except Exception:
        raw_text = ""

    # --------------------------------------------------------
    # Direct HTTP 429
    # --------------------------------------------------------

    if response.status_code == 429:
        return (
            "OpenAI API request limit has been reached. "
            "The Hybrid RAG backend is available, but "
            "LLM-based processing is temporarily unavailable. "
            "Please try again after the API usage limit resets."
        )

    # --------------------------------------------------------
    # Try JSON response
    # --------------------------------------------------------

    try:
        payload = response.json()
    except ValueError:
        payload = None

    # --------------------------------------------------------
    # FastAPI detail
    # --------------------------------------------------------

    if isinstance(payload, dict):

        detail = payload.get("detail")

        if detail:

            if isinstance(detail, list):

                messages = []

                for item in detail:

                    if isinstance(item, dict):
                        messages.append(
                            item.get(
                                "msg",
                                str(item),
                            )
                        )
                    else:
                        messages.append(
                            str(item)
                        )

                combined_detail = "; ".join(
                    messages
                )

            else:
                combined_detail = str(detail)

            if "rate limit" in combined_detail.lower():
                return (
                    "OpenAI API request limit has been reached. "
                    "The Hybrid RAG backend is available, but "
                    "LLM-based processing is temporarily unavailable. "
                    "Please try again after the API usage limit resets."
                )

            return combined_detail

        error = payload.get("error")

        if error:

            error_text = str(error)

            if "rate limit" in error_text.lower():
                return (
                    "OpenAI API request limit has been reached. "
                    "The Hybrid RAG backend is available, but "
                    "LLM-based processing is temporarily unavailable. "
                    "Please try again after the API usage limit resets."
                )

            return error_text

    # --------------------------------------------------------
    # Raw response fallback
    # --------------------------------------------------------

    if "rate limit" in raw_text.lower():
        return (
            "OpenAI API request limit has been reached. "
            "The Hybrid RAG backend is available, but "
            "LLM-based processing is temporarily unavailable. "
            "Please try again after the API usage limit resets."
        )

    return raw_text or "Unknown API error."


def get_result_error_message(result):
    """
    Convert an RAG result error into a user-friendly message.

    The /api/query endpoint can return HTTP 200 with
    success=False because RAGPipeline handles internal failures.
    """

    if not isinstance(result, dict):
        return "The Hybrid RAG backend returned an invalid response."

    error_text = result.get("error")

    if not error_text:
        return "No answer was generated."

    error_text = str(error_text)

    if "rate limit" in error_text.lower():
        return (
            "OpenAI API request limit has been reached. "
            "The Hybrid RAG backend is available, but "
            "LLM-based processing is temporarily unavailable. "
            "Please try again after the API usage limit resets."
        )

    return error_text


def check_backend():
    """
    Check whether FastAPI is reachable.
    """

    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=5,
        )

        return (
            response.status_code == 200,
            response,
        )

    except requests.RequestException:
        return False, None


def upload_pdf(uploaded_file):
    """
    Upload a PDF to the FastAPI ingestion endpoint.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }

    response = requests.post(
        UPLOAD_ENDPOINT,
        files=files,
        timeout=300,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            get_backend_error_message(response)
        )

    return response.json()


def query_backend(
    question,
    max_context_chunks=None,
):
    """
    Send a question to the FastAPI Hybrid RAG endpoint.
    """

    payload = {
        "query": question,
    }

    if max_context_chunks is not None:
        payload[
            "max_context_chunks"
        ] = max_context_chunks

    response = requests.post(
        QUERY_ENDPOINT,
        json=payload,
        timeout=300,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            get_backend_error_message(response)
        )

    return response.json()


def render_sources(sources):
    """
    Display retrieved source metadata.
    """

    if not sources:
        st.info(
            "No source chunks were returned."
        )
        return

    st.markdown(
        "### 📚 Retrieved Sources"
    )

    for index, source in enumerate(
        sources,
        start=1,
    ):

        if not isinstance(source, dict):
            source = {
                "content": str(source)
            }

        document_id = source.get(
            "document_id",
            "Unknown document",
        )

        chunk_id = source.get(
            "chunk_id",
            "Unknown chunk",
        )

        page_number = source.get(
            "page_number"
        )

        score = source.get(
            "score"
        )

        source_types = source.get(
            "source_types",
            [],
        )

        if not isinstance(
            source_types,
            list,
        ):
            source_types = []

        if source_types:
            source_label = " + ".join(
                source_types
            )
        else:
            source_label = "Unknown"

        page_label = (
            str(page_number)
            if page_number is not None
            else "N/A"
        )

        with st.expander(
            f"Source {index} • "
            f"Page {page_label} • "
            f"{source_label}"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Document ID:** "
                    f"`{document_id}`"
                )

                st.write(
                    f"**Chunk ID:** "
                    f"`{chunk_id}`"
                )

                st.write(
                    f"**Page:** "
                    f"`{page_label}`"
                )

            with col2:

                st.write(
                    f"**Retrieved by:** "
                    f"`{source_label}`"
                )

                if score is not None:

                    try:
                        score_text = (
                            f"{float(score):.4f}"
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        score_text = str(score)

                    st.write(
                        f"**Vector Score:** "
                        f"`{score_text}`"
                    )

                else:

                    st.write(
                        "**Vector Score:** "
                        "Not available"
                    )


def render_retrieval_diagnostics(result):
    """
    Display hybrid retrieval and execution diagnostics.
    """

    retrieval_summary = result.get(
        "retrieval_summary",
        {},
    )

    context_statistics = result.get(
        "context_statistics",
        {},
    )

    timings = result.get(
        "timings",
        {},
    )

    guardrail_status = result.get(
        "guardrail_status",
        "unknown",
    )

    model = result.get(
        "model",
        "Unknown",
    )

    st.markdown(
        "### 🔍 Retrieval Diagnostics"
    )

    # --------------------------------------------------------
    # Retrieval metrics
    # --------------------------------------------------------

    metric_columns = st.columns(5)

    with metric_columns[0]:

        st.metric(
            "Vector Results",
            retrieval_summary.get(
                "vector_result_count",
                0,
            ),
        )

    with metric_columns[1]:

        st.metric(
            "Graph Entities",
            retrieval_summary.get(
                "graph_entity_count",
                0,
            ),
        )

    with metric_columns[2]:

        st.metric(
            "Graph Relations",
            retrieval_summary.get(
                "graph_relationship_count",
                0,
            ),
        )

    with metric_columns[3]:

        st.metric(
            "Graph Chunks",
            retrieval_summary.get(
                "graph_chunk_count",
                0,
            ),
        )

    with metric_columns[4]:

        st.metric(
            "Combined Chunks",
            retrieval_summary.get(
                "combined_chunk_count",
                0,
            ),
        )

    # --------------------------------------------------------
    # Context metrics
    # --------------------------------------------------------

    st.markdown(
        "#### Context Statistics"
    )

    context_columns = st.columns(4)

    with context_columns[0]:

        st.metric(
            "Vector Context",
            context_statistics.get(
                "vector_context_included",
                False,
            ),
        )

    with context_columns[1]:

        st.metric(
            "Graph Context",
            context_statistics.get(
                "graph_context_included",
                False,
            ),
        )

    with context_columns[2]:

        st.metric(
            "Entities",
            context_statistics.get(
                "entity_count",
                0,
            ),
        )

    with context_columns[3]:

        st.metric(
            "Relationships",
            context_statistics.get(
                "relationship_count",
                0,
            ),
        )

    # --------------------------------------------------------
    # Timing information
    # --------------------------------------------------------

    st.markdown(
        "#### Execution Details"
    )

    execution_columns = st.columns(4)

    with execution_columns[0]:

        st.write(
            f"**Model:** `{model}`"
        )

    with execution_columns[1]:

        st.write(
            "**Total Time:** "
            f"`{format_seconds(timings.get('total_seconds'))}`"
        )

    with execution_columns[2]:

        st.write(
            "**Retrieval Time:** "
            f"`{format_seconds(timings.get('retrieval_seconds'))}`"
        )

    with execution_columns[3]:

        st.write(
            "**Generation Time:** "
            f"`{format_seconds(timings.get('answer_generation_seconds'))}`"
        )

    # --------------------------------------------------------
    # Guardrails
    # --------------------------------------------------------

    st.markdown(
        "#### Guardrail Status"
    )

    if guardrail_status == "passed":

        st.success(
            "Guardrails: Passed"
        )

    elif guardrail_status == "failed":

        st.warning(
            "Guardrails: Request blocked"
        )

    else:

        st.info(
            f"Guardrails: `{guardrail_status}`"
        )


def render_processing_summary(
    processing,
):
    """
    Display PDF ingestion statistics.
    """

    if not isinstance(
        processing,
        dict,
    ):
        return

    st.markdown(
        "#### Processing Summary"
    )

    columns = st.columns(4)

    with columns[0]:

        st.metric(
            "Chunks",
            processing.get(
                "chunk_count",
                0,
            ),
        )

    with columns[1]:

        st.metric(
            "Embeddings",
            processing.get(
                "embedding_count",
                0,
            ),
        )

    with columns[2]:

        st.metric(
            "Entities",
            processing.get(
                "entities_stored",
                0,
            ),
        )

    with columns[3]:

        st.metric(
            "Relationships",
            processing.get(
                "relationships_stored",
                0,
            ),
        )


def render_message_metadata(message):
    """
    Render sources and diagnostics stored with an assistant message.
    """

    result = message.get(
        "result"
    )

    if not result:
        return

    sources = result.get(
        "sources",
        [],
    )

    if sources:

        with st.expander(
            "📚 View Sources"
        ):

            render_sources(
                sources
            )

    if st.session_state.get(
        "show_diagnostics",
        False,
    ):

        with st.expander(
            "🔍 View Retrieval Diagnostics"
        ):

            render_retrieval_diagnostics(
                result
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🧠 Hybrid RAG Assistant"
    )

    st.markdown(
        """
        This application combines:

        - Vector retrieval with FAISS
        - Knowledge Graph retrieval with Neo4j Aura
        - Hybrid context construction
        - LLM-based answer generation
        - Guardrails
        - Source-grounded responses
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Backend status
    # --------------------------------------------------------

    st.subheader(
        "Backend Status"
    )

    backend_ready, _ = check_backend()

    if backend_ready:

        st.success(
            "FastAPI: Connected"
        )

    else:

        st.error(
            "FastAPI: Not reachable"
        )

    st.caption(
        f"Backend: {BACKEND_URL}"
    )

    if st.button(
        "🔄 Refresh Backend Status",
        use_container_width=True,
    ):
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # PDF upload
    # --------------------------------------------------------

    st.subheader(
        "📄 Upload PDF"
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help=(
            "The PDF will be processed through the "
            "complete Hybrid RAG ingestion pipeline."
        ),
    )

    if uploaded_file is not None:

        st.caption(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "Process PDF",
            use_container_width=True,
        ):

            if not backend_ready:

                st.error(
                    "FastAPI backend is not reachable. "
                    "Start FastAPI before processing a PDF."
                )

            else:

                with st.spinner(
                    "Processing PDF through Hybrid RAG pipeline..."
                ):

                    try:

                        upload_result = upload_pdf(
                            uploaded_file
                        )

                        status = upload_result.get(
                            "status"
                        )

                        # ----------------------------------------
                        # New document
                        # ----------------------------------------

                        if status == "success":

                            processing = (
                                upload_result.get(
                                    "processing",
                                    {},
                                )
                            )

                            st.session_state.current_document = {
                                "document": (
                                    upload_result.get(
                                        "document"
                                    )
                                ),
                                "processing": processing,
                            }

                            st.success(
                                "PDF processed successfully."
                            )

                            render_processing_summary(
                                processing
                            )

                        # ----------------------------------------
                        # Duplicate document
                        # ----------------------------------------

                        elif status == "duplicate":

                            st.session_state.current_document = {
                                "document": (
                                    upload_result.get(
                                        "document"
                                    )
                                ),
                                "processing": (
                                    upload_result.get(
                                        "processing",
                                        {},
                                    )
                                ),
                            }

                            st.warning(
                                upload_result.get(
                                    "message",
                                    (
                                        "This document already "
                                        "exists."
                                    ),
                                )
                            )

                            processing = (
                                upload_result.get(
                                    "processing",
                                    {},
                                )
                            )

                            if (
                                processing.get("status")
                                == "skipped"
                            ):
                                st.info(
                                    "Existing document reused. "
                                    "Downstream processing was skipped."
                                )

                        # ----------------------------------------
                        # Unexpected response
                        # ----------------------------------------

                        else:

                            st.warning(
                                upload_result.get(
                                    "message",
                                    (
                                        "Unexpected upload "
                                        "response."
                                    ),
                                )
                            )

                    except Exception as exc:

                        st.error(
                            str(exc)
                        )

    # --------------------------------------------------------
    # Current document
    # --------------------------------------------------------

    current_document = (
        st.session_state.current_document
    )

    if current_document:

        document = (
            current_document.get(
                "document"
            )
            or {}
        )

        st.divider()

        st.subheader(
            "Current Document"
        )

        st.write(
            f"**File:** "
            f"`{document.get('filename', 'Unknown')}`"
        )

        st.write(
            f"**Pages:** "
            f"`{document.get('page_count', 'N/A')}`"
        )

        st.write(
            f"**Characters:** "
            f"`{document.get('total_characters', 'N/A')}`"
        )

        document_id = document.get(
            "document_id",
            "N/A",
        )

        st.write(
            f"**Document ID:** "
            f"`{document_id}`"
        )

    st.divider()

    # --------------------------------------------------------
    # System components
    # --------------------------------------------------------

    st.subheader(
        "System Components"
    )

    st.success(
        "Vector Store: FAISS"
    )

    st.success(
        "Knowledge Graph: Neo4j Aura"
    )

    st.success(
        "Embeddings: Local Hugging Face"
    )

    st.success(
        "LLM: OpenAI"
    )

    st.success(
        "Backend: FastAPI"
    )

    st.divider()

    # --------------------------------------------------------
    # Session controls
    # --------------------------------------------------------

    st.subheader(
        "Session Controls"
    )

    st.checkbox(
        "Show retrieval diagnostics",
        value=False,
        key="show_diagnostics",
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.query_count = 0

        st.rerun()

    st.divider()

    st.metric(
        "Questions Asked",
        st.session_state.query_count,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🧠 Hybrid RAG Knowledge Assistant'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Ask questions about documents available in the Hybrid RAG
        knowledge base. Answers use vector retrieval and
        Knowledge Graph retrieval.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM STATUS
# ============================================================

if backend_ready:

    st.markdown(
        """
        <div class="status-box">
            🟢 <strong>System Status:</strong>
            FastAPI backend connected and ready.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.error(
        "FastAPI backend is not reachable. "
        "Start the backend before asking questions."
    )


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            """
            Hello! 👋

            I am your **Hybrid RAG Assistant**.

            You can:

            1. Upload a PDF from the sidebar.
            2. Ask questions about the available documents.
            3. Inspect retrieved sources.
            4. Enable retrieval diagnostics to see how
               vector and graph retrieval contributed.

            Example questions:

            - What technologies are used in the backend?
            - What technology is used for caching and sessions?
            - What is the relationship between Docker and Kubernetes?
            - Which technologies are used for observability?
            """
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )

        if role == "assistant":

            render_message_metadata(
                message
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a question about the available documents..."
)


if user_question:

    user_question = user_question.strip()

    if not user_question:

        st.warning(
            "Please enter a valid question."
        )

        st.stop()

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )

    # --------------------------------------------------------
    # Query backend
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching vector store and Knowledge Graph..."
        ):

            try:

                if not backend_ready:

                    result = {
                        "success": False,
                        "query": user_question,
                        "answer": None,
                        "sources": [],
                        "retrieval_summary": {},
                        "context_statistics": {},
                        "guardrail_status": "unknown",
                        "model": None,
                        "timings": {},
                        "debug": {},
                        "error": (
                            "FastAPI backend is not reachable."
                        ),
                    }

                else:

                    result = query_backend(
                        question=user_question
                    )

                success = bool(
                    result.get(
                        "success",
                        False,
                    )
                )

                answer = result.get(
                    "answer"
                )

                guardrail_status = result.get(
                    "guardrail_status",
                    "unknown",
                )

                # ------------------------------------------------
                # Successful answer
                # ------------------------------------------------

                if success and answer:

                    st.markdown(
                        answer
                    )

                # ------------------------------------------------
                # Guardrail rejection
                # ------------------------------------------------

                elif (
                    guardrail_status
                    == "failed"
                ):

                    friendly_message = (
                        get_result_error_message(
                            result
                        )
                    )

                    st.warning(
                        friendly_message
                    )

                # ------------------------------------------------
                # Other failure
                # ------------------------------------------------

                else:

                    friendly_message = (
                        get_result_error_message(
                            result
                        )
                    )

                    st.error(
                        friendly_message
                    )

                # ------------------------------------------------
                # Sources
                # ------------------------------------------------

                sources = result.get(
                    "sources",
                    [],
                )

                if sources:

                    with st.expander(
                        "📚 View Sources"
                    ):

                        render_sources(
                            sources
                        )

                # ------------------------------------------------
                # Diagnostics
                # ------------------------------------------------

                if st.session_state.get(
                    "show_diagnostics",
                    False,
                ):

                    with st.expander(
                        "🔍 View Retrieval Diagnostics"
                    ):

                        render_retrieval_diagnostics(
                            result
                        )

                # ------------------------------------------------
                # Store assistant response
                # ------------------------------------------------

                stored_content = (
                    answer
                    if answer
                    else get_result_error_message(
                        result
                    )
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": stored_content,
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                st.session_state.query_count += 1

            except Exception as exc:

                error_text = str(exc)

                if "rate limit" in (
                    error_text.lower()
                ):

                    display_error = (
                        "OpenAI API request limit has "
                        "been reached. The Hybrid RAG "
                        "backend is available, but "
                        "LLM-based processing is "
                        "temporarily unavailable. "
                        "Please try again after the "
                        "API usage limit resets."
                    )

                else:

                    display_error = (
                        "Unable to complete the "
                        "Hybrid RAG request."
                    )

                st.error(
                    display_error
                )

                st.caption(
                    error_text
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": display_error,
                        "result": {
                            "error": error_text,
                            "sources": [],
                            "guardrail_status": "unknown",
                        },
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                st.session_state.query_count += 1