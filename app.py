"""
app.py
Streamlit UI for the Report Summarizer.

Day 4: hardened against real-world edge cases (empty files, corrupted or
password-protected PDFs, unsupported formats, huge files) with friendly
messages instead of raw tracebacks, an optional bullet-point summary mode,
and a small styling/consistency pass.
"""

import io

import streamlit as st

from extractors import extract_text, UnsupportedFileTypeError
from summarizer import summarize, MODEL_NAME

MAX_FILE_SIZE_MB = 20


st.set_page_config(
    page_title="Report Summarizer",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
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


def to_bullets(summary_text: str) -> str:
    """
    Turn a summary paragraph into a rough bullet list by splitting on
    sentence boundaries. Good enough for a quick scan-friendly view; not a
    substitute for real extractive bullet generation.
    """
    sentences = [s.strip() for s in summary_text.replace("\n", " ").split(". ") if s.strip()]
    bullets = []
    for s in sentences:
        s = s if s.endswith(".") else s + "."
        bullets.append(f"- {s}")
    return "\n".join(bullets)


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

bullet_mode = st.sidebar.toggle(
    "Bullet-point view",
    value=False,
    help="Break the summary into bullet points instead of a paragraph.",
)

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
    # --- Edge case: empty file -------------------------------------------
    file_bytes_raw = uploaded_file.getvalue()
    if len(file_bytes_raw) == 0:
        st.error("This file is empty (0 bytes). Please upload a report that actually has content.")
        st.stop()

    # --- Edge case: file too large -----------------------------------------
    size_mb = len(file_bytes_raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"This file is {size_mb:.1f} MB, which is over the {MAX_FILE_SIZE_MB} MB "
            "limit for local processing. Try a smaller file, or split a large report "
            "into sections first."
        )
        st.stop()

    with st.spinner("Loading model (first run only, ~30s)..."):
        try:
            warm_up_model()
        except Exception as exc:
            st.error(
                "The summarization model failed to load. Make sure `torch` and "
                "`transformers` are installed correctly (see requirements.txt)."
            )
            st.caption(f"Details: {exc}")
            st.stop()

    # --- Text extraction, with friendly errors for common failure modes ---
    try:
        with st.spinner("Extracting text from your file..."):
            file_bytes = io.BytesIO(file_bytes_raw)
            report_text = extract_text(file_bytes, filename=uploaded_file.name)
    except UnsupportedFileTypeError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(
            "Couldn't read this file. It may be corrupted, password-protected, "
            "or in a format this app doesn't fully support yet."
        )
        st.caption(f"Details: {exc}")
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

    # --- Summarization, with a friendly error if generation fails ---------
    try:
        with st.spinner("Summarizing..."):
            preset = LENGTH_PRESETS[length_preset]
            summary = summarize(report_text, **preset)
    except Exception as exc:
        st.error("Something went wrong while summarizing this file. Please try again.")
        st.caption(f"Details: {exc}")
        st.stop()

    if not summary.strip():
        st.warning("The model returned an empty summary. Try a longer summary length, or a different file.")
        st.stop()

    st.divider()
    st.subheader("Summary")

    display_text = to_bullets(summary) if bullet_mode else summary
    st.markdown(display_text)

    word_count_after = len(summary.split())
    if word_count_before > 0:
        reduction_pct = round(100 * (1 - word_count_after / word_count_before))
        col1, col2, col3 = st.columns(3)
        col1.metric("Original", f"{word_count_before:,} words")
        col2.metric("Summary", f"{word_count_after:,} words")
        col3.metric("Reduction", f"{reduction_pct}%")

    st.download_button(
        label="Download summary as .txt",
        data=display_text,
        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_summary.txt",
        mime="text/plain",
    )
else:
    st.info("Upload a file above to get started. Try the samples in `samples/` if you don't have one handy.")
