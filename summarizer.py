"""
summarizer.py
Core summarization engine for the Report Summarizer app.

Uses a local Hugging Face model (t5-small by default) so the whole app
runs offline with no API key and no per-call cost.

Loads the model/tokenizer directly (AutoModelForSeq2SeqLM + generate())
rather than going through transformers' pipeline("summarization", ...)
wrapper -- the pipeline task registry has changed across transformers
versions, so calling the model directly is more robust and version-proof.

Day 1 goal: a working summarize() function that can be called on any block
of text, with automatic chunking for inputs longer than the model's token
limit. File parsing (PDF/DOCX) is handled separately in Day 2.
"""

from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# t5-small is fast and lightweight (~240MB), good for CPU-only local use.
# Swap to "facebook/bart-large-cnn" here for noticeably better quality if
# your machine can handle a slower, larger (~1.6GB) model. (bart does not
# use a task prefix -- see T5_PREFIX below.)
MODEL_NAME = "t5-small"

# t5 models expect a task prefix for summarization.
T5_PREFIX = "summarize: "

# Rough safe input size per chunk, in tokens. t5-small's encoder handles up
# to 512 tokens; we leave headroom for the task prefix and special tokens.
MAX_CHUNK_TOKENS = 450


@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_model():
    """
    Load and cache the model so it's only loaded into memory once, no
    matter how many times summarize() is called.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def _split_into_chunks(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """
    Split raw text into chunks that fit within the model's token limit.
    Splits on sentence-ish boundaries first, then packs sentences into
    chunks up to max_tokens (measured with the model's own tokenizer, not
    a word count, since tokens and words don't map 1:1).
    """
    tokenizer = _get_tokenizer()

    # crude sentence split; good enough for report-style prose
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]

    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence if sentence.endswith(".") else sentence + "."
        sentence_tokens = len(tokenizer.encode(sentence))

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _summarize_chunk(chunk: str, max_len: int, min_len: int) -> str:
    tokenizer = _get_tokenizer()
    model = _get_model()

    model_input = T5_PREFIX + chunk if "t5" in MODEL_NAME else chunk
    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    import torch

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def summarize(text: str, max_len: int = 150, min_len: int = 30) -> str:
    """
    Summarize a block of text using the local model.

    Args:
        text: raw input text (already extracted from a PDF/DOCX/TXT file).
        max_len: max tokens for each chunk's summary.
        min_len: min tokens for each chunk's summary.

    Returns:
        A single summary string. For long inputs, each chunk is summarized
        individually and the results are joined; for very long inputs you
        may want to run a second summarization pass over the combined
        chunk summaries (left as a Day 4 polish option).
    """
    text = (text or "").strip()
    if not text:
        return ""

    chunks = _split_into_chunks(text)
    summaries = [_summarize_chunk(chunk, max_len, min_len) for chunk in chunks]

    return " ".join(summaries)


if __name__ == "__main__":
    sample_text = """
    Artificial intelligence adoption in the enterprise has accelerated
    significantly over the past two years. Organizations across finance,
    healthcare, and retail are integrating machine learning models into
    core workflows, ranging from fraud detection to personalized customer
    recommendations. Despite this growth, many companies report challenges
    around data quality, model interpretability, and regulatory compliance.
    A recent industry survey found that 68 percent of enterprises consider
    data governance their top barrier to scaling AI initiatives, while only
    23 percent have a dedicated AI ethics review process in place. Investment
    in AI infrastructure is expected to continue rising, with cloud providers
    reporting record demand for GPU compute resources dedicated to model
    training and inference workloads.
    """

    print("Loading model and summarizing sample text...\n")
    output = summarize(sample_text, max_len=60, min_len=15)
    print("SUMMARY:")
    print(output)
