import streamlit as st
import sqlite3
import uuid
import json
import pathlib
import tempfile
import os
from datetime import datetime

# ── Page config (must be first Streamlit command) ────────────────────────────
st.set_page_config(
    page_title="Thoth",
    page_icon="𓁟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Heavy imports behind a loading indicator ─────────────────────────────────
with st.spinner("Loading models. Please Wait…"):
    from threads import _list_threads, _save_thread_meta, _delete_thread, checkpointer, DB_PATH
    from rag import rag_graph_compiled
    from documents import (
        load_processed_files,
        load_and_vectorize_document,
        reset_vector_store,
        DocumentLoader,
    )
    from models import (
        get_current_model,
        set_model,
        list_all_models,
        list_local_models,
        is_model_local,
        pull_model,
        DEFAULT_MODEL,
    )

# ── Ensure default model is available ────────────────────────────────────────
if not is_model_local(DEFAULT_MODEL):
    with st.status(f"Downloading default model **{DEFAULT_MODEL}**…", expanded=True) as status:
        for progress in pull_model(DEFAULT_MODEL):
            total = int(progress.get("total", 0) if progress.get("total") else 0)
            completed = int(progress.get("completed", 0) if progress.get("completed") else 0)
            msg = progress.get("status", "")
            if total:
                pct = int(completed / total * 100)
                status.update(label=f"Downloading {DEFAULT_MODEL}: {pct}%")
            else:
                status.update(label=f"{DEFAULT_MODEL}: {msg}")
        status.update(label=f"✅ {DEFAULT_MODEL} ready!", state="complete")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 5rem; }

    /* Pin chat input to bottom, spanning the chat column width */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 1rem;
        width: calc(75% - 6rem);
        z-index: 100;
    }

    /* Thread buttons */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        border-radius: 8px;
    }

    /* Active thread highlight */
    .active-thread {
        background-color: #262730;
        border-radius: 8px;
        padding: 4px 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: load chat history from checkpoint ───────────────────────────────
def load_thread_messages(thread_id: str) -> list[dict]:
    """Return list of {'role': ..., 'content': ...} from the checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = rag_graph_compiled.get_state(config)
        if snapshot and snapshot.values and "messages" in snapshot.values:
            msgs = []
            for m in snapshot.values["messages"]:
                role = "user" if m.type == "human" else "assistant"
                msgs.append({"role": role, "content": m.content})
            return msgs
    except Exception:
        pass
    return []


# ── Session state defaults ──────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
    st.session_state.thread_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "search_documents" not in st.session_state:
    st.session_state.search_documents = True
if "search_wikipedia" not in st.session_state:
    st.session_state.search_wikipedia = True
if "search_arxiv" not in st.session_state:
    st.session_state.search_arxiv = True
if "search_web" not in st.session_state:
    st.session_state.search_web = True

if "current_model" not in st.session_state:
    st.session_state.current_model = get_current_model()

# Always sync module-level model state from session state (survives reruns & refreshes)
if get_current_model() != st.session_state.current_model:
    set_model(st.session_state.current_model)


# ═════════════════════════════════════════════════════════════════════════════
# LEFT SIDEBAR – Thread Manager
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("𓁟 Thoth")
    st.caption("God of Wisdom, Writing, and Knowledge\nYour Private Knowledge Agent")
    st.divider()

    # New thread button
    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        tid = uuid.uuid4().hex[:12]
        name = f"Thread {datetime.now().strftime('%b %d, %H:%M')}"
        _save_thread_meta(tid, name)
        st.session_state.thread_id = tid
        st.session_state.thread_name = name
        st.session_state.messages = []
        st.rerun()

    st.markdown("#### Conversations")

    threads = _list_threads()
    if not threads:
        st.info("No conversations yet.")
    for tid, name, created, updated in threads:
        is_active = tid == st.session_state.thread_id
        col_thread, col_del = st.columns([5, 1])
        with col_thread:
            label = f"{'\u25b8 ' if is_active else ''}{name}"
            if st.button(
                label,
                key=f"thread_{tid}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state.thread_id = tid
                st.session_state.thread_name = name
                st.session_state.messages = load_thread_messages(tid)
                st.rerun()
        with col_del:
            if st.button("\U0001f5d1", key=f"del_thread_{tid}", help=f"Delete {name}"):
                _delete_thread(tid)
                if st.session_state.thread_id == tid:
                    st.session_state.thread_id = None
                    st.session_state.thread_name = None
                    st.session_state.messages = []
                st.rerun()

    # ── Settings popover pinned to bottom of sidebar ─────────────────────────
    st.markdown(
        """<div style="position: fixed; bottom: 1rem; width: inherit; z-index: 200;">""",
        unsafe_allow_html=True,
    )
    with st.popover("⚙️ Settings", use_container_width=True):
        st.markdown("#### Model")
        all_models = list_all_models()
        local_models = list_local_models()
        current = st.session_state.current_model
        if current not in all_models:
            all_models = sorted(set(all_models + [current]))
        idx = all_models.index(current) if current in all_models else 0

        selected_model = st.selectbox(
            "Select model",
            options=all_models,
            index=idx,
            format_func=lambda m: f"{'✅' if m in local_models else '⬇️'}  {m}",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("#### Retrieval Sources")
        st.session_state.search_documents = st.toggle(
            "📄 Documents", value=st.session_state.search_documents, key="toggle_documents"
        )
        st.session_state.search_wikipedia = st.toggle(
            "🌐 Wikipedia", value=st.session_state.search_wikipedia, key="toggle_wikipedia"
        )
        st.session_state.search_arxiv = st.toggle(
            "📚 Arxiv", value=st.session_state.search_arxiv, key="toggle_arxiv"
        )
        st.session_state.search_web = st.toggle(
            "🔍 Web Search", value=st.session_state.search_web, key="toggle_web"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle model switch ──────────────────────────────────────────────────
    if selected_model and selected_model != st.session_state.current_model:
        if not is_model_local(selected_model):
            with st.status(f"Downloading **{selected_model}**…", expanded=True) as status:
                for progress in pull_model(selected_model):
                    total = int(progress.get("total", 0) if progress.get("total") else 0)
                    completed = int(progress.get("completed", 0) if progress.get("completed") else 0)
                    msg = progress.get("status", "")
                    if total:
                        pct = int(completed / total * 100)
                        status.update(label=f"Downloading {selected_model}: {pct}%")
                    else:
                        status.update(label=f"{selected_model}: {msg}")
                status.update(label=f"✅ {selected_model} ready!", state="complete")
        set_model(selected_model)
        st.session_state.current_model = selected_model
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA – Chat
# ═════════════════════════════════════════════════════════════════════════════
chat_col, doc_col = st.columns([3, 1], gap="large")

with chat_col:
    if st.session_state.thread_id is None:
        st.markdown(
            """
            <div style='text-align:center; padding-top: 8rem;'>
                <h1>𓁟 Thoth</h1>
                <p style='font-size:1.15rem; color: #888;'>
                    Select a conversation from the sidebar or start a new one.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"### 💬 {st.session_state.thread_name}")

        # Render existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("Ask a question…"):
            # Show user message immediately
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Auto-rename thread on first user message
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            if len(user_msgs) == 1:
                auto_name = user_input[:50].rstrip()
                if len(user_input) > 50:
                    auto_name += "…"
                st.session_state.thread_name = auto_name
                _save_thread_meta(st.session_state.thread_id, auto_name)

            # Invoke RAG graph
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            _save_thread_meta(
                st.session_state.thread_id, st.session_state.thread_name
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    result = rag_graph_compiled.invoke(
                        {
                            "messages": [("human", user_input)],
                            "search_documents": st.session_state.search_documents,
                            "search_wikipedia": st.session_state.search_wikipedia,
                            "search_arxiv": st.session_state.search_arxiv,
                            "search_web": st.session_state.search_web,
                        },
                        config=config,
                    )
                    answer = result["answer"].content

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL – Document Manager
# ═════════════════════════════════════════════════════════════════════════════
with doc_col:
    st.markdown("### 📄 Documents")

    # ── Upload ───────────────────────────────────────────────────────────────
    supported_exts = list(DocumentLoader.supported_file_types.keys())
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[ext.lstrip(".") for ext in supported_exts],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        for uf in uploaded_files:
            suffix = pathlib.Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir="."
            ) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner(f"Processing {uf.name}\u2026"):
                    load_and_vectorize_document(
                        tmp_path, skip_if_processed=False, display_name=uf.name
                    )
                st.success(f"\u2714 {uf.name}")
            except Exception as exc:
                st.error(f"Failed to process {uf.name}: {exc}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()

    # ── List processed documents ─────────────────────────────────────────────
    st.markdown("**Indexed documents**")
    processed = load_processed_files()
    if processed:
        for fp in sorted(processed):
            name = pathlib.Path(fp).name
            st.markdown(f"- 📎 {name}")
        if st.button("\U0001f5d1 Clear all documents", use_container_width=True):
            reset_vector_store()
            st.session_state.uploader_key += 1
            st.rerun()
    else:
        st.caption("No documents indexed yet.")
