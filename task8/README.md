# Logic-LM with LangChain + Prolog

A Logic-LM–style pipeline that combines **LangChain**, **RAG**, and a **pure-Python Prolog inference engine** to answer natural-language questions with logical deduction traces.

---

## Architecture

```
Natural Language Question
        │
        ▼
  RAG Retriever  ──► relevant KB facts/rules (FAISS vector store)
        │
        ▼
  LLM Translator (GPT-4o-mini via LangChain)
        │  "Is tweety an animal?" → is_a(tweety, animal)
        ▼
  Prolog Engine (pure Python, backward chaining)
        │  backward-chain over knowledge_base.pl
        ▼
  (True/False  +  inference trace)
        │
        ▼
  LLM Explainer (GPT-4o-mini via LangChain)
        │
        ▼
  Human-readable explanation
```

Inspired by:
- **Logic-LM** (Pan et al., 2023) — LLM → symbolic solver → result
- **LINC** (Olausson et al., 2023) — neurosymbolic reasoning via Prolog
- **Symbol-LLM** (Xu et al., 2023) — symbolic knowledge injection into LLMs

---

## Project Structure

```
langchain_logic/
├── knowledge_base.pl   # Prolog KB: ~15 facts, ~8 rules
├── prolog_engine.py    # Pure-Python backward-chaining Prolog
├── main.py             # Full LangChain + RAG + LLM pipeline
├── test_engine.py      # Offline tests (no API key needed)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone / create the project folder

```bash
mkdir langchain_logic && cd langchain_logic
# copy all files here
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI key

```bash
cp .env.example .env
# edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 5. Run offline tests (no API key needed)

```bash
python test_engine.py
```

Expected: **10/10 passed**

### 6. Run the full LangChain pipeline

```bash
python main.py
```

---

## Knowledge Base Overview

**Individuals:** tweety, sam, rex, whiskers, nemo, goldie, leo, bella, koko

**Facts (15):** `is_a/2`, `has_property/2`, `eats/2`

**Rules (8):**
- Penguin → Bird → Animal (taxonomy chain)
- Dog/Cat/Fish/Lion/Parrot → Animal
- `is_pet/1` — dogs and cats
- `is_carnivore/1` — eats meat or fish
- `is_herbivore/1` — eats plants or seeds
- `is_domestic/1` — pets + parrots
- `is_flightless/1` — bird + `\+` can_fly (negation-as-failure)

---

## Sample Output

```
❓  Question : Is sam flightless?
🔁  Prolog goal : is_flightless(sam)
🧠  Result      : TRUE ✓
📝  Trace:
    TRY: is_flightless(sam) ← is_a(sam, bird), not(has_property(sam, can_fly))
      TRY: is_a(sam, bird) ← is_a(sam, penguin)
        FACT: is_a(sam, penguin)
          NAF: not(has_property(sam, can_fly)) holds (inner goal failed)
    ✓ Query PROVED: is_flightless(sam)
```

---

## Requirements

- Python 3.11+
- OpenAI API key (for `main.py` only; `test_engine.py` is free)
