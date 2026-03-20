"""
Task 9 — LangGraph Migration
Migrates Task 8 LangChain RAG + Logic-LM pipeline to LangGraph
with a relevancy judge and self-refinement loop.

Architecture (LangGraph nodes):
  retrieve → judge_relevance → [refine | translate] → solve → explain

Run:
    python main.py

Requirements (install first):
    pip install langgraph langchain langchain-openai langchain-community \
                faiss-cpu openai python-dotenv
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

from prolog_engine import PrologEngine

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE BASE  (same solar-system KB from Task 4)
# ─────────────────────────────────────────────────────────────────────────────
KB_TEXT = """
% Facts
planet(mercury, terrestrial, 0, no).
planet(venus, terrestrial, 0, no).
planet(earth, terrestrial, 1, no).
planet(mars, terrestrial, 2, no).
planet(jupiter, gas_giant, 95, yes).
planet(saturn, gas_giant, 146, yes).
planet(uranus, ice_giant, 28, yes).
planet(neptune, ice_giant, 16, yes).

orbits_sun(mercury, 1). orbits_sun(venus, 2). orbits_sun(earth, 3).
orbits_sun(mars, 4).    orbits_sun(jupiter, 5). orbits_sun(saturn, 6).
orbits_sun(uranus, 7).  orbits_sun(neptune, 8).

diameter_km(mercury, 4879).  diameter_km(venus, 12104).
diameter_km(earth, 12742).   diameter_km(mars, 6779).
diameter_km(jupiter, 139820). diameter_km(saturn, 116460).
diameter_km(uranus, 50724).  diameter_km(neptune, 49244).

% Rules
inner_planet(X) :- orbits_sun(X, N), N =< 4.
outer_planet(X) :- orbits_sun(X, N), N > 4.
large_planet(X) :- diameter_km(X, D), D > 50000.
has_rings(X) :- planet(X, _, _, yes).
has_moons(X) :- planet(X, _, M, _), M > 0.
neighbors(X, Y) :- orbits_sun(X, Nx), orbits_sun(Y, Ny), Ny is Nx + 1.
neighbors(X, Y) :- orbits_sun(X, Nx), orbits_sun(Y, Ny), Ny is Nx - 1.
habitable_candidate(X) :- inner_planet(X), has_moons(X).
gas_or_ice(X) :- planet(X, gas_giant, _, _).
gas_or_ice(X) :- planet(X, ice_giant, _, _).
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR STORE  (RAG index built line-by-line)
# ─────────────────────────────────────────────────────────────────────────────
def build_vectorstore():
    lines = [l.strip() for l in KB_TEXT.strip().splitlines() if l.strip() and not l.startswith("%")]
    docs = [Document(page_content=line) for line in lines]
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str           # natural language query
    context: str            # retrieved KB chunks
    prolog_goal: str        # translated Prolog goal
    result: bool            # solver result
    trace: list             # inference trace
    answer: str             # final natural-language explanation
    relevance: str          # "relevant" | "irrelevant"
    refinement_round: int   # tracks self-refinement attempts
    error: str              # solver error if any

# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM
# ─────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. NODES
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """RAG: retrieve top-5 KB chunks relevant to the question."""
    vs = build_vectorstore()
    docs = vs.similarity_search(state["question"], k=5)
    context = "\n".join(d.page_content for d in docs)
    return {**state, "context": context, "refinement_round": 0, "error": ""}


def judge_relevance(state: GraphState) -> GraphState:
    """LangGraph judge: decide if retrieved context is relevant."""
    prompt = f"""You are a relevancy judge for a Prolog knowledge base.
Question: {state["question"]}
Retrieved context:
{state["context"]}

Is this context relevant and sufficient to answer the question with Prolog?
Reply with EXACTLY one word: relevant OR irrelevant."""
    resp = llm.invoke(prompt)
    verdict = resp.content.strip().lower()
    if "irrelevant" in verdict:
        return {**state, "relevance": "irrelevant"}
    return {**state, "relevance": "relevant"}


def refine_retrieval(state: GraphState) -> GraphState:
    """Self-refinement: broaden query and retrieve again with extra context."""
    round_num = state["refinement_round"] + 1
    print(f"  [refine] Round {round_num}: context was irrelevant — broadening retrieval...")
    vs = build_vectorstore()
    # Try fetching more chunks (k=8) with a rephrased query
    rephrased = llm.invoke(
        f"Rephrase this question using synonyms to improve Prolog KB retrieval:\n{state['question']}"
    ).content.strip()
    docs = vs.similarity_search(rephrased, k=8)
    context = "\n".join(d.page_content for d in docs)
    return {**state, "context": context, "refinement_round": round_num}


def translate_to_prolog(state: GraphState) -> GraphState:
    """LLM translator: convert NL question + context into a Prolog goal."""
    prompt = f"""You are a Prolog translator. Given context from a knowledge base and a question,
output ONLY the Prolog goal (no explanation, no backticks, no period).

Context:
{state["context"]}

Question: {state["question"]}

Examples:
- "Which planets are large?" → large_planet(X)
- "Is Mars an inner planet?" → inner_planet(mars)
- "Does Saturn have rings?" → has_rings(saturn)
- "Which planets are neighbors of Earth?" → neighbors(earth, X)

Prolog goal:"""
    resp = llm.invoke(prompt)
    goal = resp.content.strip().rstrip(".")
    return {**state, "prolog_goal": goal}


def solve(state: GraphState) -> GraphState:
    """Run the Prolog backward-chaining engine."""
    engine = PrologEngine()
    # Load all KB facts/rules from the text
    for line in KB_TEXT.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("%"):
            engine.load_line(line)

    goal_str = state["prolog_goal"]
    print(f"  [solve] Goal: {goal_str}")
    try:
        solutions, trace = engine.query(goal_str)
        result = len(solutions) > 0
        return {**state, "result": result, "trace": trace, "error": ""}
    except Exception as e:
        return {**state, "result": False, "trace": [], "error": str(e)}


def self_refine_goal(state: GraphState) -> GraphState:
    """If solver failed with an error, ask LLM to fix the goal."""
    round_num = state["refinement_round"] + 1
    print(f"  [self-refine] Round {round_num}: fixing goal due to error: {state['error']}")
    prompt = f"""The Prolog goal "{state['prolog_goal']}" caused this error: {state['error']}
Available predicates: planet/4, orbits_sun/2, diameter_km/2, inner_planet/1, outer_planet/1,
large_planet/1, has_rings/1, has_moons/1, neighbors/2, habitable_candidate/1, gas_or_ice/1.
Fix the Prolog goal. Output ONLY the corrected goal (no punctuation):"""
    fixed_goal = llm.invoke(prompt).content.strip().rstrip(".")
    return {**state, "prolog_goal": fixed_goal, "refinement_round": round_num}


def explain(state: GraphState) -> GraphState:
    """LLM explainer: convert raw result + trace into plain English."""
    trace_str = "\n".join(state["trace"]) if state["trace"] else "(no trace)"
    prompt = f"""Summarize this Prolog query result in 2-3 plain English sentences.
Question: {state["question"]}
Goal: {state["prolog_goal"]}
Result: {"TRUE" if state["result"] else "FALSE"}
Trace:
{trace_str}"""
    answer = llm.invoke(prompt).content.strip()
    return {**state, "answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_relevance(state: GraphState) -> Literal["refine_retrieval", "translate_to_prolog"]:
    if state["relevance"] == "irrelevant" and state["refinement_round"] < 2:
        return "refine_retrieval"
    return "translate_to_prolog"


def route_solver(state: GraphState) -> Literal["self_refine_goal", "explain"]:
    if state["error"] and state["refinement_round"] < 2:
        return "self_refine_goal"
    return "explain"


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve)
    g.add_node("judge_relevance", judge_relevance)
    g.add_node("refine_retrieval", refine_retrieval)
    g.add_node("translate_to_prolog", translate_to_prolog)
    g.add_node("solve", solve)
    g.add_node("self_refine_goal", self_refine_goal)
    g.add_node("explain", explain)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "judge_relevance")

    g.add_conditional_edges(
        "judge_relevance",
        route_relevance,
        {
            "refine_retrieval": "refine_retrieval",
            "translate_to_prolog": "translate_to_prolog",
        }
    )

    # After refine, re-judge
    g.add_edge("refine_retrieval", "judge_relevance")

    g.add_edge("translate_to_prolog", "solve")

    g.add_conditional_edges(
        "solve",
        route_solver,
        {
            "self_refine_goal": "self_refine_goal",
            "explain": "explain",
        }
    )

    # After self-refine, re-run solver
    g.add_edge("self_refine_goal", "solve")

    g.add_edge("explain", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 8. RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

QUERIES = [
    "Which planets are large?",
    "Is Mars an inner planet?",
    "Does Saturn have rings?",
    "Which planets are neighbors of Earth?",
    "Which planets have moons?",
    "Is Jupiter a gas or ice planet?",
    "Which planets could be habitable candidates?",
]

if __name__ == "__main__":
    graph = build_graph()
    print("=" * 60)
    print("Task 9 — LangGraph RAG + Relevancy Judge + Self-Refinement")
    print("=" * 60)

    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}] {q}")
        state = graph.invoke({"question": q})
        print(f"  Goal:     {state['prolog_goal']}")
        print(f"  Result:   {'TRUE' if state['result'] else 'FALSE'}")
        print(f"  Relevant: {state['relevance']}")
        if state["trace"]:
            print(f"  Trace:    {state['trace'][0]}")
        print(f"  Answer:   {state['answer']}")

    # --- Self-refinement demo: intentional typo ---
    print("\n" + "=" * 60)
    print("Self-Refinement Demo (intentional typo: 'large_planett')")
    print("=" * 60)
    demo_state: GraphState = {
        "question": "Which planets are large?",
        "context": "large_planet(X) :- diameter_km(X, D), D > 50000.",
        "prolog_goal": "large_planett(X)",   # typo on purpose
        "result": False,
        "trace": [],
        "answer": "",
        "relevance": "relevant",
        "refinement_round": 0,
        "error": "",
    }

    # Build a mini-graph: just solve → self_refine_goal → solve → explain
    mini = StateGraph(GraphState)
    mini.add_node("solve", solve)
    mini.add_node("self_refine_goal", self_refine_goal)
    mini.add_node("explain", explain)
    mini.set_entry_point("solve")
    mini.add_conditional_edges("solve", route_solver,
        {"self_refine_goal": "self_refine_goal", "explain": "explain"})
    mini.add_edge("self_refine_goal", "solve")
    mini.add_edge("explain", END)
    mini_graph = mini.compile()

    out = mini_graph.invoke(demo_state)
    print(f"  Fixed Goal: {out['prolog_goal']}")
    print(f"  Result:     {'TRUE' if out['result'] else 'FALSE'}")
    print(f"  Answer:     {out['answer']}")
