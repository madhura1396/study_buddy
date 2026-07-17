"""Streamlit UI for grounded Q&A over your study materials.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st

from src.drive import MissingCredentialsError, download_file, list_importable_files
from src.generate import GenerationError, MissingAPIKeyError, generate_answer
from src.ingest import get_collection, run_ingestion

st.set_page_config(page_title="Study Buddy", page_icon="📚")
st.title("📚 Study Buddy")
st.caption("Ask questions grounded in your ingested study materials.")


@st.cache_data(ttl=30)
def load_ingested_chapters() -> dict[str, list[str]]:
    """Return {source_filename: [section headings...]} for everything ingested."""
    metadatas = get_collection().get()["metadatas"]
    chapters: dict[str, list[str]] = {}
    for meta in sorted(metadatas, key=lambda m: (m["source"], m["chunk_index"])):
        headings = chapters.setdefault(meta["source"], [])
        heading = meta.get("heading", "")
        if heading and (not headings or headings[-1] != heading):
            headings.append(heading)
    return chapters


with st.sidebar:
    st.header("Uploaded chapters")
    chapters = load_ingested_chapters()
    if not chapters:
        st.info("No documents ingested yet. Run `python -m src.ingest` after adding files to data/sample_docs/.")
    else:
        for source, headings in chapters.items():
            with st.expander(source):
                if headings:
                    for h in headings:
                        st.markdown(f"- {h}")
                else:
                    st.caption("(no section headings detected)")

    st.divider()
    st.header("Import from Google Drive")
    drive_search = st.text_input("Search by file name", key="drive_search")
    if st.button("Search Drive"):
        try:
            st.session_state.drive_results = list_importable_files(drive_search)
        except MissingCredentialsError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Google Drive authentication/search failed: {e}")

    drive_results = st.session_state.get("drive_results", [])
    if drive_results:
        selected_names = st.multiselect(
            "Files found",
            options=[f["name"] for f in drive_results],
        )
        if st.button("Import selected", disabled=not selected_names):
            with st.spinner("Downloading and ingesting..."):
                for f in drive_results:
                    if f["name"] in selected_names:
                        download_file(f)
                run_ingestion()
            st.session_state.drive_results = []
            load_ingested_chapters.clear()
            st.success(f"Imported {len(selected_names)} file(s).")
            st.rerun()
    elif "drive_results" in st.session_state:
        st.caption("No matching files found.")

# Chat-style history in session_state: st.text_input previously left the
# question un-cleared after answering, and Streamlit only reruns a script
# when a widget's *value* changes — so re-submitting an unedited question,
# or wanting to ask a fresh one without manually clearing the box, silently
# did nothing. st.chat_input auto-clears after every submission and always
# triggers a rerun, which is what makes "ask the next question" work.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for i, hit in message.get("hits", []):
            label = f"[{i}] {hit['source']} (chunk {hit['chunk_index']})"
            if hit["heading"]:
                label += f" — {hit['heading']}"
            with st.expander(label):
                st.text(hit["text"])

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
        for i, hit in numbered_hits:
            label = f"[{i}] {hit['source']} (chunk {hit['chunk_index']})"
            if hit["heading"]:
                label += f" — {hit['heading']}"
            with st.expander(label):
                st.text(hit["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "hits": numbered_hits}
    )
