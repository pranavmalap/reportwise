"""
test_pipeline.py
Day 2 end-to-end check: file -> extracted text -> summary.

Run this after activating your venv:
    python test_pipeline.py

It runs every file in samples/ through extract_text() and summarize(),
so you can eyeball the extraction + summarization quality on realistic
report formats before wiring up the Streamlit UI in Day 3.
"""

from pathlib import Path

from extractors import extract_text
from summarizer import summarize

SAMPLES_DIR = Path(__file__).parent / "samples"


def run():
    sample_files = sorted(SAMPLES_DIR.glob("*"))
    if not sample_files:
        print(f"No sample files found in {SAMPLES_DIR}")
        return

    for path in sample_files:
        if path.suffix.lower() not in (".pdf", ".docx", ".txt"):
            continue

        print(f"\n{'=' * 70}")
        print(f"FILE: {path.name}")
        print("=" * 70)

        text = extract_text(path)
        print(f"Extracted {len(text)} characters, {len(text.split())} words.\n")

        summary = summarize(text, max_len=80, min_len=20)
        print("SUMMARY:")
        print(summary)


if __name__ == "__main__":
    run()
