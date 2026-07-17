"""Streamlit UI for grounded Q&A over your study materials.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st

from src.generate import GenerationError, MissingAPIKeyError, generate_answer

st.set_page_config(page_title="Study Buddy", page_icon="📚")
st.title("📚 Study Buddy")
st.caption("Ask questions grounded in your ingested study materials.")

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
