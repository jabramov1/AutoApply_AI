# AutoApply AI

AutoApply AI is a multi-agent LangChain + Streamlit app that turns a resume into structured data, critiques it, matches it against sample job postings, and generates tailored cover letters.

## Features

- Upload a resume (`.pdf`, `.txt`, `.md`)
- Parse resume into structured JSON
- Run a simple, local ATS-style check (no extra API calls)
- Critique the resume with a recruiter-style rubric
- Match the resume against mock job postings and rank results
- Generate a personalized cover letter per job and download it

## Tech Stack

- Streamlit UI
- LangChain + OpenAI-compatible chat model (`langchain-openai`)
- PDF text extraction via `pypdf` (through `PyPDFLoader`)

## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure API access

This project expects an OpenAI-compatible API key in `API_KEY`.

- Option A: Create a `.env` file in the project root:

```bash
API_KEY=your_key_here
```

- Option B: Export an environment variable:

```bash
export API_KEY=your_key_here
```

Notes:
- `app.py` sets `OPENAI_BASE_URL` to `https://api.ai.it.cornell.edu` (Cornell endpoint).
- `app.py` maps `API_KEY` → `OPENAI_API_KEY` at runtime.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Project Notes / Limitations

- Job postings are mock data defined in `app.py` (`MOCK_JOBS`).
- The resume parser/critic/matcher/cover-letter steps rely on LLM outputs and may be imperfect; always review results.
- Avoid uploading sensitive information unless you understand and accept where the API is hosted and how data is handled.

## File Overview

- `app.py`: Streamlit app + all agents (parser, critic, matcher, cover letter)
- `requirements.txt`: Python dependencies
