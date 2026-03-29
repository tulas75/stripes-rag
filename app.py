"""Stripes RAG — Streamlit Chat App with smolagents."""

from __future__ import annotations

from pathlib import Path

import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stripes RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Available models (same list that was in sa.py)
# ---------------------------------------------------------------------------
MODELS = {
    "Qwen3.5 4B (Ollama)": "ollama_chat/qwen3.5:4b-q8_0",
    "Qwen3.5 9B (Ollama)": "ollama_chat/qwen3.5:9b-q8_0",
    "Qwen3.5 0.8B (Ollama)": "ollama_chat/qwen3.5:0.8b",
    "Qwen3.5 2B (Ollama)": "ollama_chat/qwen3.5:2b",
    "DeepSeek Chat": "deepseek/deepseek-chat",
    "Mistral Small 3.2": "deepinfra/mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Mistral Small": "deepinfra/mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen3 Next 80B": "deepinfra/Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen3 30B": "deepinfra/Qwen/Qwen3-30B-A3B",
    "MiniMax M2.1": "anthropic/MiniMax-M2.1",
    "Qwen3.5 35B (Ollama)": "ollama_chat/qwen3.5:35b",
    "Qwen3 4B (Ollama)": "ollama_chat/qwen3:4b-instruct-2507-q8_0",
    "Groq Llama 3.1 8B": "groq/llama-3.1-8b-instant",
    "Mistral Small (API)": "mistral/mistral-small-latest",
    "Qwen3.5 0.8B (llama.cpp)": ("openai/Qwen3.5-0.8B-Q8_0", "http://192.168.1.18:8082/v1", "dummy"),

}

LANGUAGES = {"Italiano": "ITA", "English": "ENG"}

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Prompt profile
    from stripes_rag.prompts import list_profiles, get_profile

    profile_names = list_profiles()
    selected_profile_name = st.selectbox(
        "Prompt Profile", profile_names, index=1,  # default: Project Architect
    )
    profile = get_profile(selected_profile_name)
    st.caption(f"*{profile.description}*")

    st.divider()

    selected_model = st.selectbox("Model", list(MODELS.keys()), index=0)
    _model_entry = MODELS[selected_model]
    if isinstance(_model_entry, tuple):
        model_id = _model_entry[0]
        model_api_base = _model_entry[1]
        model_api_key = _model_entry[2] if len(_model_entry) > 2 else None
    else:
        model_id, model_api_base, model_api_key = _model_entry, None, None

    selected_lang = st.selectbox("Language", list(LANGUAGES.keys()), index=0)
    language = LANGUAGES[selected_lang]

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    k_results = st.slider("Retrieval depth (k)", 3, 10, 5)
    max_steps = st.slider("Agent max steps", 2, 10, 6)

    st.divider()

    # Reranker toggle
    from stripes_rag.reranker import is_reranker_available
    _reranker_available = is_reranker_available()
    use_reranker = st.checkbox(
        "Reranker",
        value=_reranker_available,
        disabled=not _reranker_available,
        help="Cross-encoder reranking (TEI or LiteLLM)" if _reranker_available
             else "Set RERANKER_PROVIDER or RERANKER_URL to enable",
    )

    st.divider()
    st.caption("Stripes RAG v0.1")


# ---------------------------------------------------------------------------
# Cached resources — loaded once, reused across reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model & vectorstore…")
def load_vectorstore():
    from stripes_rag.db import get_engine, get_vectorstore
    from stripes_rag.embeddings import get_embeddings

    engine = get_engine()
    embeddings = get_embeddings()
    return get_vectorstore(engine, embeddings)


vectorstore = load_vectorstore()

# ---------------------------------------------------------------------------
# RetrieverTool (same as sa.py, but k is dynamic)
# ---------------------------------------------------------------------------
from smolagents import Tool


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Searches the organizational knowledge base for relevant past projects, "
        "experiences, methodologies, and lessons learned. Call this tool multiple "
        "times with different queries to gather broad context from different angles "
        "(e.g., technical approaches, stakeholder concerns, similar past projects, "
        "risks encountered)."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query. Use varied queries to explore different facets of the topic.",
        }
    }
    output_type = "string"

    def __init__(self, vectorstore, k: int = 10, use_reranker: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
        self.k = k
        self.use_reranker = use_reranker
        self.retrieved_chunks = []
        self.retrieval_time: float = 0.0
        self.retrieval_calls: int = 0

    def forward(self, query: str) -> str:
        import time as _time

        from stripes_rag.config import settings as _settings
        from stripes_rag.reranker import rerank

        assert isinstance(query, str), "Your search query must be a string"

        k_fetch = self.k * _settings.reranker_top_k_multiplier if self.use_reranker else self.k

        _t0 = _time.perf_counter()
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k_fetch)

        reranked = False
        if self.use_reranker:
            for doc, distance in docs_with_scores:
                doc.metadata['_vector_distance'] = distance
            docs_with_scores, reranked = rerank(query, docs_with_scores, top_k=self.k)

        self.retrieval_time += _time.perf_counter() - _t0
        self.retrieval_calls += 1

        result = "\nRetrieved documents:\n"
        for i, (doc, score) in enumerate(docs_with_scores):
            vector_sim = 1 - doc.metadata.pop('_vector_distance', 1 - score) if reranked else 1 - score
            reranker_score = score if reranked else None
            similarity = reranker_score if reranker_score is not None else vector_sim
            source = doc.metadata.get("source_file", "unknown")
            headings = doc.metadata.get("headings", "")
            pages = doc.metadata.get("page_numbers", "")

            self.retrieved_chunks.append({
                "source": source,
                "similarity": vector_sim,
                "reranker_score": reranker_score,
                "headings": headings,
                "pages": pages,
                "content": doc.page_content
            })

            result += (
                f"\n\n===== Document {i} (similarity: {similarity:.4f}) =====\n"
                f"Source: {source}\n"
                f"Similarity: {similarity:.4f}\n"
            )
            if headings:
                result += f"Headings: {headings}\n"
            if pages:
                result += f"Pages: {pages}\n"
            result += f"\n{doc.page_content}"

        return result


def _split_follow_ups(text: str) -> tuple[str, list[str]]:
    """Split answer text from follow-up questions section.

    Returns (answer_body, list_of_follow_up_strings).
    """
    pattern = r"##\s*Follow[\-\s]?up\s+Questions?"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return text.strip(), []

    body = text[:match.start()].strip()
    follow_up_section = text[match.end():]
    follow_ups = [
        line.lstrip("-*• ").strip()
        for line in follow_up_section.splitlines()
        if line.strip() and line.strip().lstrip("-*• ")
    ]
    return body, follow_ups


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# UI: title + chat
# ---------------------------------------------------------------------------
st.title("📚 Stripes RAG")
st.caption(f"{profile.name} — {profile.description}")

# Render existing chat history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander("📚 Sources & References"):
                for chunk in msg["chunks"]:
                    cols = st.columns([5, 1])
                    ref = f"**{chunk['source']}**"
                    if chunk['pages']:
                        ref += f" (Page {chunk['pages']})"
                    cols[0].markdown(f"- {ref}")
                    with cols[1].popover("Read"):
                        st.markdown(f"**Source:** {chunk['source']}")
                        if chunk['pages']:
                            st.markdown(f"**Pages:** {chunk['pages']}")
                        if chunk['headings']:
                            st.markdown(f"**Headings:** {chunk['headings']}")
                        if chunk.get("reranker_score") is not None:
                            st.caption(f"Reranker: {chunk['reranker_score']:.4f} · Similarity: {chunk['similarity']:.4f}")
                        else:
                            st.caption(f"Similarity: {chunk['similarity']:.4f}")
                        st.markdown("---")
                        st.markdown(chunk['content'])
        if msg.get("follow_ups"):
            st.markdown("**Follow-up questions:**")
            for fq_idx, fq in enumerate(msg["follow_ups"]):
                if st.button(fq, key=f"fq_{msg_idx}_{fq_idx}"):
                    st.session_state.pending_follow_up = fq
        if msg.get("stats"):
            st.caption(msg["stats"])

# Handle follow-up button clicks
if "pending_follow_up" in st.session_state:
    _follow_up = st.session_state.pop("pending_follow_up")
    st.session_state.messages.append({"role": "user", "content": _follow_up})
    st.rerun()

# Determine if we need to run the agent
prompt = st.chat_input("Ask a question about your indexed documents…")

# Check if last message is an unanswered user message (from follow-up click)
needs_response = (
    not prompt
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and (len(st.session_state.messages) < 2 or st.session_state.messages[-2]["role"] != "user")
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

if prompt or needs_response:
    query = prompt or st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base & synthesizing…"):
            import time as _time

            from smolagents import CodeAgent, LiteLLMModel

            retriever_tool = RetrieverTool(vectorstore, k=k_results, use_reranker=use_reranker)

            agent = CodeAgent(
                tools=[retriever_tool],
                #model=LiteLLMModel(model_id=model_id, temperature=temperature, reasoning_effort="low"),
                model=LiteLLMModel(model_id=model_id, api_base=model_api_base, api_key=model_api_key, temperature=temperature),
                max_steps=max_steps,
                stream_outputs=False,
                additional_authorized_imports=["json"],
                verbosity_level=0,
            )

            t_start = _time.perf_counter()
            try:
                answer = agent.run(
                    query,
                    additional_args=dict(
                        additional_notes=profile.template.format(language=language)
                    ),
                )

                # Capture, deduplicate, and sort by similarity descending
                unique_chunks = []
                seen_contents = set()
                for c in retriever_tool.retrieved_chunks:
                    if c["content"] not in seen_contents:
                        seen_contents.add(c["content"])
                        unique_chunks.append(c)
                if any(c.get("reranker_score") is not None for c in unique_chunks):
                    unique_chunks.sort(key=lambda c: c["reranker_score"], reverse=True)
                else:
                    unique_chunks.sort(key=lambda c: c["similarity"], reverse=True)

            except Exception as e:
                answer = f"❌ **Error:** {e}"
                unique_chunks = []

            elapsed = _time.perf_counter() - t_start
            token_usage = agent.monitor.get_total_token_counts()

        body, follow_ups = _split_follow_ups(str(answer))
        st.markdown(body)
        if unique_chunks:
            with st.expander("📚 Sources & References"):
                for chunk in unique_chunks:
                    cols = st.columns([5, 1])
                    ref = f"**{chunk['source']}**"
                    if chunk['pages']:
                        ref += f" (Page {chunk['pages']})"
                    cols[0].markdown(f"- {ref}")
                    with cols[1].popover("Read"):
                        st.markdown(f"**Source:** {chunk['source']}")
                        if chunk['pages']:
                            st.markdown(f"**Pages:** {chunk['pages']}")
                        if chunk['headings']:
                            st.markdown(f"**Headings:** {chunk['headings']}")
                        if chunk.get("reranker_score") is not None:
                            st.caption(f"Reranker: {chunk['reranker_score']:.4f} · Similarity: {chunk['similarity']:.4f}")
                        else:
                            st.caption(f"Similarity: {chunk['similarity']:.4f}")
                        st.markdown("---")
                        st.markdown(chunk['content'])
        if follow_ups:
            st.markdown("**Follow-up questions:**")
            for fq_idx, fq in enumerate(follow_ups):
                if st.button(fq, key=f"fq_new_{fq_idx}"):
                    st.session_state.pending_follow_up = fq

        retr_time = retriever_tool.retrieval_time
        retr_calls = retriever_tool.retrieval_calls
        llm_time = elapsed - retr_time
        stats_text = (
            f"⏱ {elapsed:.1f}s · "
            f"🔍 {retr_time:.1f}s retrieval ({retr_calls} call{'s' if retr_calls != 1 else ''}) · "
            f"🤖 {llm_time:.1f}s LLM · "
            f"⬆ {token_usage.input_tokens:,} in · "
            f"⬇ {token_usage.output_tokens:,} out"
        )
        st.caption(stats_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": body,
        "chunks": unique_chunks,
        "follow_ups": follow_ups,
        "stats": stats_text,
    })
