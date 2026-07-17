"""Streamlit UI for grounded Q&A over your study materials.

Run from the project root:
    streamlit run app.py
"""

from datetime import date, datetime
from pathlib import Path

import streamlit as st

from src.config import DATA_DIR
from src.drive import (
    FOLDER_MIME_TYPE,
    MissingCredentialsError,
    download_file,
    get_cached_drive_link,
    get_or_create_drive_link,
    list_importable_files,
    resolve_to_files,
)
from src.generate import GenerationError, MissingAPIKeyError, generate_answer
from src.ingest import (
    UNCATEGORIZED,
    delete_document,
    get_collection,
    import_local_folder,
    run_ingestion,
)

st.set_page_config(page_title="Study Buddy", page_icon="📚")
st.title("📚 Study Buddy")


@st.cache_data(ttl=30)
def load_ingested_docs() -> list[dict]:
    """Return one entry per ingested source file: {source, category,
    uploaded_at, headings}, newest-uploaded first."""
    metadatas = get_collection().get()["metadatas"]
    docs: dict[tuple[str, str], dict] = {}
    for meta in sorted(metadatas, key=lambda m: m["chunk_index"]):
        key = (meta["category"], meta["source"])
        doc = docs.setdefault(
            key,
            {
                "source": meta["source"],
                "category": meta["category"],
                "uploaded_at": meta["uploaded_at"],
                "headings": [],
            },
        )
        heading = meta.get("heading", "")
        if heading and (not doc["headings"] or doc["headings"][-1] != heading):
            doc["headings"].append(heading)
    return sorted(docs.values(), key=lambda d: d["uploaded_at"], reverse=True)


def render_sources(numbered_hits: list[tuple[int, dict]]) -> None:
    """Render all source chunks for one answer inside a single collapsed
    expander, rather than one expander per chunk — a wall of always-visible
    boxes under every answer felt cluttered and un-chat-like."""
    if not numbered_hits:
        return
    with st.expander(f"Show sources ({len(numbered_hits)})"):
        for i, hit in numbered_hits:
            label = f"[{i}] {hit['source']} (chunk {hit['chunk_index']})"
            if hit["heading"]:
                label += f" — {hit['heading']}"
            st.markdown(f"**{label}**")
            st.text(hit["text"])
            st.divider()


def local_file_path(category: str, source: str) -> Path:
    if category == UNCATEGORIZED:
        return DATA_DIR / source
    return DATA_DIR / category / source


def existing_categories() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


def category_picker(key_prefix: str) -> str:
    """A selectbox of existing category folders plus a "create new" option;
    an empty new-category name falls back to today's date, so files always
    land somewhere sensible even if you don't bother naming a folder."""
    options = [*existing_categories(), "+ Create new folder..."]
    choice = st.selectbox("Save into folder", options, key=f"{key_prefix}_cat_choice")
    if choice == "+ Create new folder...":
        new_name = st.text_input("New folder name", key=f"{key_prefix}_cat_new")
        return new_name.strip() or date.today().isoformat()
    return choice


ask_tab, materials_tab, import_tab = st.tabs(["💬 Ask", "📂 Study Materials", "📥 Import"])

with ask_tab:
    top_col, button_col = st.columns([5, 1])
    with top_col:
        st.caption("Ask questions grounded in your ingested study materials.")
    with button_col:
        if st.button("🆕 New chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat-style history in session_state: st.text_input previously left the
    # question un-cleared after answering, and Streamlit only reruns a script
    # when a widget's *value* changes — so re-submitting an unedited question,
    # or wanting to ask a fresh one without manually clearing the box, silently
    # did nothing. st.chat_input auto-clears after every submission and always
    # triggers a rerun, which is what makes "ask the next question" work.
    # New questions always append below the existing thread; only the "New
    # chat" button above clears it, so past Q&A stays visible as you go.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("hits", []))

    question = st.chat_input("Ask a question, e.g. What is the linearity assumption?")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    result = generate_answer(question)
                except MissingAPIKeyError as e:
                    st.error(str(e))
                    st.stop()
                except GenerationError as e:
                    st.error(str(e))
                    st.stop()

            st.markdown(result["answer"])
            numbered_hits = list(enumerate(result["hits"], start=1))
            render_sources(numbered_hits)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "hits": numbered_hits}
        )

with materials_tab:
    docs = load_ingested_docs()
    if not docs:
        st.info(
            "No documents ingested yet. Head to the Import tab to add some."
        )
    else:
        st.subheader("Recently uploaded")
        for doc in docs[:5]:
            uploaded = datetime.fromtimestamp(doc["uploaded_at"]).strftime("%b %d, %Y %I:%M %p")
            st.markdown(f"**{doc['source']}** — _{doc['category']}_ — uploaded {uploaded}")

        st.divider()
        st.subheader("By folder")
        by_category: dict[str, list[dict]] = {}
        for doc in docs:
            by_category.setdefault(doc["category"], []).append(doc)

        # Each folder is its own (collapsed-by-default) expander; files
        # inside are plain listings rather than nested expanders, since
        # Streamlit doesn't support nesting an expander inside another.
        for category in sorted(by_category, key=lambda c: (c == UNCATEGORIZED, c)):
            category_docs = by_category[category]
            with st.expander(f"📁 {category} ({len(category_docs)})"):
                for doc in category_docs:
                    st.markdown(f"**{doc['source']}**")
                    if doc["headings"]:
                        for h in doc["headings"]:
                            st.markdown(f"- {h}")
                    else:
                        st.caption("(no section headings detected)")
                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        cached_url = get_cached_drive_link(category, doc["source"])
                        if cached_url:
                            st.link_button("🔗 Open in Google Docs", cached_url)
                        elif st.button(
                            "⬆️ Upload & open in Google Docs",
                            key=f"drive_link_{category}_{doc['source']}",
                        ):
                            with st.spinner("Uploading to Google Drive..."):
                                url = get_or_create_drive_link(
                                    local_file_path(category, doc["source"]), category
                                )
                            st.link_button("🔗 Open in Google Docs", url)
                    with action_col2:
                        if st.button("🗑️ Delete", key=f"delete_{category}_{doc['source']}"):
                            delete_document(category, doc["source"])
                            load_ingested_docs.clear()
                            st.success(f"Deleted {doc['source']}.")
                            st.rerun()
                    st.divider()

with import_tab:
    st.subheader("Import from Google Drive")
    drive_search = st.text_input("Search by file or folder name", key="drive_search")
    if st.button("Search Drive"):
        try:
            st.session_state.drive_results = list_importable_files(drive_search)
        except MissingCredentialsError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Google Drive authentication/search failed: {e}")

    drive_results = st.session_state.get("drive_results", [])
    if drive_results:

        def _label(f: dict) -> str:
            return f"📁 {f['name']}" if f["mimeType"] == FOLDER_MIME_TYPE else f["name"]

        labels_by_name = {_label(f): f for f in drive_results}
        selected_labels = st.multiselect(
            "Files & folders found (a folder imports everything inside it, "
            "using the folder's name as the category)",
            options=list(labels_by_name.keys()),
        )
        selected_entries = [labels_by_name[label] for label in selected_labels]
        any_individual_files = any(e["mimeType"] != FOLDER_MIME_TYPE for e in selected_entries)
        fallback_category = category_picker("drive") if any_individual_files else None

        if st.button("Import selected", disabled=not selected_labels):
            with st.spinner("Resolving folders, downloading, and ingesting..."):
                resolved = resolve_to_files(selected_entries)
                for f, category in resolved:
                    dest_dir = DATA_DIR / (category or fallback_category)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    download_file(f, dest_dir=dest_dir)
                run_ingestion()
            st.session_state.drive_results = []
            load_ingested_docs.clear()
            st.success(f"Imported {len(resolved)} file(s).")
            st.rerun()
    elif "drive_results" in st.session_state:
        st.caption("No matching files found.")

    st.divider()
    st.subheader("Upload from your computer")
    uploaded_files = st.file_uploader(
        "Upload .txt or .docx files", type=["txt", "docx"], accept_multiple_files=True
    )
    upload_category = category_picker("upload") if uploaded_files else None
    if uploaded_files and st.button("Ingest uploaded files"):
        with st.spinner("Saving and ingesting..."):
            dest_dir = DATA_DIR / upload_category
            dest_dir.mkdir(parents=True, exist_ok=True)
            for uploaded in uploaded_files:
                (dest_dir / uploaded.name).write_bytes(uploaded.getvalue())
            run_ingestion()
        load_ingested_docs.clear()
        st.success(f"Ingested {len(uploaded_files)} file(s) into '{upload_category}'.")
        st.rerun()

    st.divider()
    st.subheader("Import a local folder")
    st.caption(
        "Point at a folder on this computer, e.g. one containing subfolders "
        "like 'linear regression', 'logistic', 'kmeans' — each subfolder is "
        "imported as its own category, exactly as it's laid out on disk."
    )
    folder_path = st.text_input("Folder path", placeholder="/Users/you/ml", key="local_folder_path")
    if st.button("Import folder", disabled=not folder_path):
        try:
            with st.spinner("Copying files and ingesting..."):
                copied = import_local_folder(Path(folder_path).expanduser())
                run_ingestion()
            load_ingested_docs.clear()
            st.success(f"Imported {copied} file(s).")
            st.rerun()
        except (NotADirectoryError, FileNotFoundError) as e:
            st.error(str(e))
