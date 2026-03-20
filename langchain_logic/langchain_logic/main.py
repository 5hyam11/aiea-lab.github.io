"""
main.py
Logic-LM style pipeline using LangChain + a pure-Python Prolog engine.

Architecture (inspired by Logic-LM, LINC, Symbol-LLM):
  1. RAG: embed the KB facts → retrieve relevant context for the query
  2. LLM Translator: natural-language question → Prolog goal string
  3. Prolog Engine: backward-chain solve → (True/False, trace)
  4. LLM Explainer: turn the trace into a human-readable explanation

Requirements (pip install):
  langchain langchain-openai langchain-community faiss-cpu openai python-dotenv
"""

import os
import textwrap
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from prolog_engine import load_engine, PrologEngine

load_dotenv()

# ──────────────────────────────────────────────
# 1.  BUILD RAG INDEX from KB facts
# ──────────────────────────────────────────────

KB_PATH = "knowledge_base.pl"

def kb_to_documents(kb_path: str) -> list[Document]:
    """Convert each non-comment, non-empty line into a Document."""
    docs = []
    with open(kb_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            docs.append(Document(page_content=line, metadata={"source": kb_path}))
    return docs


def build_rag_retriever(kb_path: str):
    docs = kb_to_documents(kb_path)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


# ──────────────────────────────────────────────
# 2.  LLM-BASED QUERY TRANSLATOR (Logic-LM style)
# ──────────────────────────────────────────────

TRANSLATE_SYSTEM = textwrap.dedent("""\
    You are a Prolog expert.  Given:
      - A natural-language question
      - Relevant facts/rules from a Prolog knowledge base

    Output ONLY a single valid Prolog goal (no explanation, no punctuation at the end).
    The goal must use predicates that actually appear in the provided facts/rules.

    Examples:
      Question: "Is tweety an animal?"
      Prolog goal: is_a(tweety, animal)

      Question: "Can sam fly?"
      Prolog goal: has_property(sam, can_fly)

      Question: "Is rex a carnivore?"
      Prolog goal: is_carnivore(rex)
""")

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", TRANSLATE_SYSTEM),
    ("human", "KB context:\n{context}\n\nQuestion: {question}"),
])


# ──────────────────────────────────────────────
# 3.  LLM-BASED TRACE EXPLAINER
# ──────────────────────────────────────────────

EXPLAIN_SYSTEM = textwrap.dedent("""\
    You are a helpful assistant that explains logical deductions in plain English.
    Given a Prolog inference trace (a list of reasoning steps), write a concise
    2–4 sentence explanation suitable for a student.  Do not repeat the raw Prolog syntax;
    describe what was proved or disproved and why.
""")

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", EXPLAIN_SYSTEM),
    ("human", "Query: {query}\nResult: {result}\nTrace:\n{trace}"),
])


# ──────────────────────────────────────────────
# 4.  FULL PIPELINE
# ──────────────────────────────────────────────

class LogicLMPipeline:
    def __init__(self, kb_path: str = KB_PATH):
        print("🔧  Loading knowledge base …")
        self.engine: PrologEngine = load_engine(kb_path)

        print("🔧  Building RAG index …")
        self.retriever = build_rag_retriever(kb_path)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        parser = StrOutputParser()

        self.translate_chain = translate_prompt | llm | parser
        self.explain_chain   = explain_prompt  | llm | parser

    def run(self, question: str) -> dict:
        print(f"\n{'='*60}")
        print(f"❓  Question : {question}")

        # Step 1 – RAG retrieval
        docs = self.retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)
        print(f"\n📄  RAG context ({len(docs)} docs):\n{context}")

        # Step 2 – Translate to Prolog goal
        prolog_goal = self.translate_chain.invoke(
            {"question": question, "context": context}
        ).strip().rstrip('.')
        print(f"\n🔁  Prolog goal : {prolog_goal}")

        # Step 3 – Prolog inference
        success, trace = self.engine.query(prolog_goal)
        result_str = "TRUE ✓" if success else "FALSE ✗"
        print(f"\n🧠  Result      : {result_str}")
        print("📝  Trace:")
        for step in trace:
            print(f"    {step}")

        # Step 4 – Natural-language explanation
        explanation = self.explain_chain.invoke({
            "query":  prolog_goal,
            "result": result_str,
            "trace":  "\n".join(trace),
        })
        print(f"\n💬  Explanation : {explanation}")
        print('='*60)

        return {
            "question":    question,
            "context":     context,
            "prolog_goal": prolog_goal,
            "success":     success,
            "trace":       trace,
            "explanation": explanation,
        }


# ──────────────────────────────────────────────
# 5.  DEMO
# ──────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Is tweety an animal?",
    "Can sam fly?",
    "Is rex a carnivore?",
    "Is whiskers a pet?",
    "Is sam flightless?",
    "Does bella eat meat?",
    "Is nemo an animal?",
]

if __name__ == "__main__":
    pipeline = LogicLMPipeline()
    for q in DEMO_QUESTIONS:
        pipeline.run(q)
