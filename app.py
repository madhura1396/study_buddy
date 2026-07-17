"""Streamlit UI for grounded Q&A over your study materials.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st

from src.generate import GenerationError, MissingAPIKeyError, generate_answer
from src.ingest import get_collection

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

question = st.text_input("Ask a question", placeholder="e.g. What is the linearity assumption?")

if question:
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

    if result["sources"]:
        st.subheader("Sources")
        for i, hit in enumerate(result["hits"], start=1):
            label = f"[{i}] {hit['source']} (chunk {hit['chunk_index']})"
            if hit["heading"]:
                label += f" — {hit['heading']}"
            with st.expander(label):
                st.text(hit["text"])
