# Implementation Report: Logic-LM with LangChain and Prolog

**GitHub Repository:** https://github.com/YOUR_USERNAME/langchain-logic-lm  
*(replace with your actual repo URL after pushing)*

---

## Overview

This project implements a **Logic-LM–style neurosymbolic reasoning pipeline** using LangChain, a FAISS vector store for RAG, and a custom pure-Python Prolog backward-chaining engine. The system accepts natural-language questions, retrieves relevant logical context, translates the question into a Prolog goal via an LLM, runs formal inference, and returns a boolean answer with a full deduction trace.

---

## Architecture

The pipeline has four stages, directly inspired by Logic-LM (Pan et al., 2023), LINC (Olausson et al., 2023), and Symbol-LLM (Xu et al., 2023):

1. **RAG Retrieval** — The Prolog knowledge base is embedded line-by-line using OpenAI embeddings into a FAISS vector store. For each query, the top-5 most semantically relevant facts/rules are retrieved and injected as context.

2. **LLM Translator** — A LangChain `ChatPromptTemplate` chain backed by GPT-4o-mini translates the natural-language question plus the retrieved KB context into a single valid Prolog goal string (e.g., `is_flightless(sam)`).

3. **Prolog Inference Engine** — A pure-Python engine (`prolog_engine.py`) parses the `.pl` knowledge base and performs backward chaining. It supports facts, Horn-clause rules, unification, variable renaming, and negation-as-failure (`\+`). The engine yields a boolean result and a step-by-step inference trace.

4. **LLM Explainer** — A second LangChain chain takes the Prolog trace and explains it in plain English for the user.

---

## Knowledge Base

The knowledge base (`knowledge_base.pl`) models an animal domain with **15 facts** and **8 rules**. Facts assert category membership (`is_a`), properties (`has_property`), and diet (`eats`). Rules encode taxonomy chains (penguin → bird → animal), derived predicates (`is_carnivore`, `is_herbivore`, `is_pet`, `is_domestic`), and a negation-as-failure rule (`is_flightless`).

---

## Tests Performed

The offline test suite (`test_engine.py`) ran **10 test cases** covering all rule types:

| Query | Expected | Result |
|---|---|---|
| `is_a(tweety, animal)` | TRUE | **PASS** |
| `is_a(sam, bird)` | TRUE | **PASS** |
| `has_property(sam, can_fly)` | FALSE | **PASS** |
| `is_flightless(sam)` | TRUE | **PASS** |
| `is_carnivore(rex)` | TRUE | **PASS** |
| `is_carnivore(tweety)` | FALSE | **PASS** |
| `is_pet(whiskers)` | TRUE | **PASS** |
| `is_domestic(koko)` | TRUE | **PASS** |
| `is_a(nemo, animal)` | TRUE | **PASS** |
| `is_herbivore(nemo)` | TRUE | **PASS** |

**10/10 tests passed.** The most notable test is `is_flightless(sam)`: it requires chaining through the penguin→bird rule and then applying negation-as-failure to confirm no `can_fly` property exists — a multi-step deduction that exercises the full engine.

The full `main.py` pipeline additionally tested 7 natural-language questions through the complete LangChain + RAG + LLM path, with all queries correctly translated and solved.

---

## Key Design Decisions

- **Pure-Python Prolog** was chosen over SWI-Prolog to avoid OS-level dependencies and to make the trace introspectable in Python.
- **RAG on the KB** ensures the LLM translator receives only relevant facts, reducing hallucination of non-existent predicates.
- **Separation of translation and inference** mirrors the Logic-LM paper's principle: the LLM handles language, the symbolic engine handles reasoning — neither role bleeds into the other.

---

*All code is available in the GitHub repository linked above.*
