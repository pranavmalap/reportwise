# 📄 Report Summarizer

A local, offline AI app that turns PDF and Word reports into clean, short
summaries — no API key, no per-call cost, and no data ever leaves your
machine. Built with [Streamlit](https://streamlit.io) and a local
Hugging Face model ([`t5-small`](https://huggingface.co/t5-small)).

**🔗 Live demo:** _add your deployed link here once deployed (see below)_

## Why

Reading through long reports to pull out the key points is slow. This app
extracts the text from a PDF/DOCX/TXT report, chunks it to fit the model's
input limit, and returns a short summary in seconds — entirely on-device.

## Features

- Upload a report as **PDF**, **Word (.docx)**, or plain **.txt**
- Adjustable summary length: Short / Medium / Long
- Optional bullet-point view
- Word-count reduction stats (e.g. "1,200 words → 140 words, 88% shorter")
- Friendly error handling for empty files, oversized files, corrupted or
  password-protected PDFs, and unsupported formats
- Download the summary as a `.txt` file
- Runs **fully offline** after the first model download — no API key,
  no cost per summary

## How it works

1. **Extraction** (`extractors.py`) — pulls raw text out of the uploaded
   file (including table contents for DOCX), then strips common report
   noise: standalone page numbers, "CONFIDENTIAL" headers/footers, broken
   line breaks, and excess whitespace.
2. **Chunking** (`summarizer.py`) — `t5-small` can only handle ~512 tokens
   at a time, so long reports are split on sentence boundaries into chunks
   that fit, without cutting a sentence in half.
3. **Summarization** (`summarizer.py`) — each chunk is summarized with
   beam search (`model.generate()`) and the chunk summaries are joined
   into the final result. The model and tokenizer are loaded once and
   cached (`@st.cache_resource`), so the UI stays fast after the first
   request.
4. **UI** (`app.py`) — Streamlit handles upload, settings, and display,
   with every stage wrapped in error handling so failures show a clear
   message instead of a crash.

## Tech stack

| Layer          | Choice                                   |
|----------------|-------------------------------------------|
| UI             | Streamlit                                 |
| Model          | `t5-small` (Hugging Face `transformers`)  |
| PDF parsing    | `pypdf`                                   |
| DOCX parsing   | `python-docx`                             |
| Cost           | $0 — fully local inference, no API key    |

## Running it locally

```bash
git clone https://github.com/pranavmalap/reportwise.git
cd reportwise
python -m venv venv
venv\Scripts\activate        # on Windows
# source venv/bin/activate   # on macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads `t5-small` (~240MB) from Hugging Face — after
that, everything runs offline.

Don't have a report handy? Try any file in the `samples/` folder.

## Deploying (free)

This app is small enough to deploy for free on either platform below.

**Streamlit Community Cloud** (recommended, easiest):
1. Push this repo to GitHub (already done if you're reading this from
   the repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
3. Click **New app**, pick this repo, branch `main`, and set the main
   file to `app.py`.
4. Deploy. The first build takes a few minutes while it installs
   `torch`/`transformers` and downloads the model on first use.

**Hugging Face Spaces** (alternative, more headroom for ML apps):
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   choosing the **Streamlit** SDK.
2. Push this repo's contents to the Space's git remote (Spaces work
   exactly like a GitHub repo).
3. The Space builds and serves the app automatically.

## Project structure

```
.
├── app.py              # Streamlit UI
├── summarizer.py        # Model loading, chunking, summarization
├── extractors.py         # PDF/DOCX/TXT text extraction + cleanup
├── test_pipeline.py       # End-to-end test: file -> text -> summary
├── samples/               # Sample reports (PDF/DOCX/TXT) for testing
└── requirements.txt
```

## Known limitations

- Scanned PDFs (an image of a document rather than real text) aren't
  supported — there's no OCR step yet.
- `t5-small` favors speed over summary quality. Swapping `MODEL_NAME` in
  `summarizer.py` to `facebook/bart-large-cnn` gives noticeably better
  summaries at the cost of a larger download and slower inference.

## License

MIT — see [LICENSE](LICENSE).
