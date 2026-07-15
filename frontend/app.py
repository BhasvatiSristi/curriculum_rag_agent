"""
app.py

Streamlit frontend for the Curriculum RAG Assistant.
"""

import requests
import streamlit as st

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Curriculum RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

# ----------------------------------------------------------
# Session State
# ----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------
# API Helper Functions
# ----------------------------------------------------------

def check_api():
    """
    Check whether the FastAPI backend is running.
    """
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )

        return response.json()

    except Exception:
        return None


def get_sources():
    """
    Fetch indexed PDF sources.
    """
    try:
        response = requests.get(
            f"{API_URL}/sources",
            timeout=5,
        )

        return response.json()["sources"]

    except Exception:
        return []


def ask_question(question: str, mode: str):
    """
    Send a question to the backend.
    """
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "question": question,
            "mode": mode,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

with st.sidebar:

    st.title("🎓 Curriculum RAG")

    health = check_api()

    if health:

        st.success("Backend Connected")

        st.metric(
            "Indexed Chunks",
            health["indexed_chunks"],
        )

    else:

        st.error("Backend Offline")

    st.divider()

    st.subheader("Indexed Documents")

    sources = get_sources()

    if sources:

        for source in sources:
            st.write(f"📄 {source}")

    else:
        st.caption("No documents indexed.")

    st.divider()

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        [
            "hybrid",
            "dense",
            "bm25",
        ],
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------
# Main Page
# ----------------------------------------------------------

st.title("🎓 IIITDM Curriculum Assistant")

st.caption(
    "Ask questions about the curriculum using Retrieval-Augmented Generation (RAG)."
)

# ----------------------------------------------------------
# Display Chat History
# ----------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ----------------------------------------------------------
# Chat Input
# ----------------------------------------------------------

question = st.chat_input(
    "Ask a question about the curriculum..."
)

# ----------------------------------------------------------
# Handle User Input
# ----------------------------------------------------------

if question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching curriculum..."):

            try:

                response = ask_question(
                    question,
                    retrieval_mode,
                )

                answer = response["answer"]
                sources = response["sources"]

                st.markdown(answer)

                # Show sources
                if sources:

                    with st.expander("📚 Sources Used"):

                        for source in sources:

                            st.markdown(
                                f"""
**Document:** {source['source']}

**Page:** {source['page']}
"""
                            )

            except requests.exceptions.ConnectionError:

                answer = (
                    "❌ Unable to connect to the backend.\n\n"
                    "Make sure FastAPI is running."
                )

                st.error(answer)

            except Exception as e:

                answer = f"❌ {e}"

                st.error(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )