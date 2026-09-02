"""
app.py
Streamlit UI for the Report Summarizer.

Day 3 goal: upload a PDF/DOCX/TXT report, adjust summary length in the
sidebar, see the summary, and download it as a .txt file. The model is
cached with @st.cache_resource so it only loads once per server process,
not on every rerun.
"""

import io

import streamlit as st

from extractors import extract_text
from summarizer import summarize, MODEL_NAME


st.set_page_config(
    page_title="Report Summarizer",
    page_icon="📄",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Cached model warm-up
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def warm_up_model():
    """
    Trigger model + tokenizer loading once and cache it for the life of the
    server process, so the first real summarization request isn't the one
    that pays the (slow) model-loading cost.
    """
    from summarizer import _get_model, _get_tokenizer  # noqa: E402

    _get_tokenizer()
    _get_model()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Summary settings")

length_preset = st.sidebar.radio(
    "Summary length",
    options=["Short", "Medium", "Long"],
    index=1,
    help="Controls how long each summarized chunk of the report is.",
)

LENGTH_PRESETS = {
    "Short": dict(max_len=60, min_len=15),
    "Medium": dict(max_len=120, min_len=30),
    "Long": dict(max_len=200, min_len=60),
}

st.sidebar.caption(f"Model: `{MODEL_NAME}` (runs locally, no API key needed)")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("📄 Report Summarizer")
st.write(
    "Upload a report as PDF, Word (.docx), or plain text, and get a clean "
    "summary back in seconds -- everything runs locally on your machine."
)

uploaded_file = st.file_uploader(
    "Upload a report",
    type=["pdf", "docx", "txt"],
)

if uploaded_file is not None:
    with st.spinner("Loading model (first run only, ~30s)..."):
        warm_up_model()

    try:
        with st.spinner("Extracting text from your file..."):
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            report_text = extract_text(file_bytes, filename=uploaded_file.name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not report_text.strip():
        st.warning(
            "No readable text was found in this file. If it's a scanned "
            "PDF (an image of a document rather than real text), this app "
            "can't extract it yet -- try a text-based PDF or DOCX instead."
        )
        st.stop()

    word_count_before = len(report_text.split())
    st.caption(f"Extracted {word_count_before:,} words from **{uploaded_file.name}**.")

    with st.expander("View extracted text"):
        st.text(report_text)

    with st.spinner("Summarizing..."):
        preset = LENGTH_PRESETS[length_preset]
        summary = summarize(report_text, **preset)

    st.subheader("Summary")
    st.write(summary)

    word_count_after = len(summary.split())
    if word_count_before > 0:
        reduction_pct = round(100 * (1 - word_count_after / word_count_before))
        st.caption(
            f"{word_count_before:,} words → {word_count_after:,} words "
            f"({reduction_pct}% shorter)"
        )

    st.download_button(
        label="Download summary as .txt",
        data=summary,
        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_summary.txt",
        mime="text/plain",
    )
else:
    st.info("Upload a file above to get started. Try the samples in `samples/` if you don't have one handy.")
