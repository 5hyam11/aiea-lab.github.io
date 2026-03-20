"""
prolog_engine.py
A pure-Python backward-chaining Prolog interpreter.
Supports: facts, rules with :-, negation-as-failure (\\+), unification.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Generator, List, Optional, Tuple


# ─────────────────────────── Term Types ───────────────────────────

class Var:
    """A Prolog variable (starts with uppercase or _)."""
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name
    def __hash__(self):
        return hash(self.name)


class Atom:
    """A Prolog atom (lowercase string or quoted)."""
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name
    def __hash__(self):
        return hash(self.name)


class Compound:
    """A Prolog compound term: functor(arg1, arg2, ...)."""
    def __init__(self, functor: str, args: List[Any]):
        self.functor = functor
        self.args = args
    def __repr__(self):
        return f"{self.functor}({', '.join(map(repr, self.args))})"
    def __eq__(self, other):
        return (isinstance(other, Compound) and
                self.functor == other.functor and self.args == other.args)
    def __hash__(self):
        return hash((self.functor, tuple(self.args)))


Term = Atom | Var | Compound


# ─────────────────────────── Parser ───────────────────────────

def tokenize(text: str) -> List[str]:
    text = re.sub(r'%[^\n]*', '', text)           # strip comments
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = re.findall(r'\\[+]|:-|[A-Za-z_][A-Za-z0-9_]*|\'[^\']*\'|[(),.]', text)
    return tokens


def parse_term(tokens: List[str], pos: int) -> Tuple[Term, int]:
    tok = tokens[pos]
    # Negation-as-failure: \+ Goal
    if tok in ('\\+', '\\\\+'):
        goal, pos = parse_term(tokens, pos + 1)
        return Compound('naf', [goal]), pos
    # Atom or compound
    if re.match(r'^[a-z_]', tok) or (tok.startswith("'") and tok.endswith("'")):
        name = tok.strip("'")
        if pos + 1 < len(tokens) and tokens[pos + 1] == '(':
            pos += 2  # skip name and '('
            args = []
            while tokens[pos] != ')':
                if tokens[pos] == ',':
                    pos += 1
                    continue
                arg, pos = parse_term(tokens, pos)
                args.append(arg)
            return Compound(name, args), pos + 1  # skip ')'
        return Atom(name), pos + 1
    # Variable
    if re.match(r'^[A-Z_]', tok):
        return Var(tok), pos + 1
    raise ValueError(f"Unexpected token: {tok!r} at position {pos}")


def parse_clause(tokens: List[str], pos: int) -> Tuple[Tuple, int]:
    """Returns (head, body_list) and new pos."""
    head, pos = parse_term(tokens, pos)
    if pos < len(tokens) and tokens[pos] == ':-':
        pos += 1  # skip :-
        body = []
        while pos < len(tokens) and tokens[pos] != '.':
            if tokens[pos] == ',':
                pos += 1
                continue
            goal, pos = parse_term(tokens, pos)
            body.append(goal)
        return (head, body), pos + 1  # skip '.'
    else:
        return (head, []), pos + 1   # fact


def parse_kb(text: str) -> List[Tuple]:
    tokens = tokenize(text)
    clauses = []
    pos = 0
    while pos < len(tokens):
        clause, pos = parse_clause(tokens, pos)
        clauses.append(clause)
    return clauses


# ─────────────────────────── Unification ───────────────────────────

Env = Dict[str, Any]


def walk(term: Term, env: Env) -> Term:
    while isinstance(term, Var) and term.name in env:
        term = env[term.name]
    return term


def unify(t1: Term, t2: Term, env: Env) -> Optional[Env]:
    t1 = walk(t1, env)
    t2 = walk(t2, env)
    if t1 == t2:
        return env
    if isinstance(t1, Var):
        return {**env, t1.name: t2}
    if isinstance(t2, Var):
        return {**env, t2.name: t1}
    if (isinstance(t1, Compound) and isinstance(t2, Compound) and
            t1.functor == t2.functor and len(t1.args) == len(t2.args)):
        for a, b in zip(t1.args, t2.args):
            env = unify(a, b, env)
            if env is None:
                return None
        return env
    return None


def substitute(term: Term, env: Env) -> Term:
    term = walk(term, env)
    if isinstance(term, Compound):
        return Compound(term.functor, [substitute(a, env) for a in term.args])
    return term


# ─────────────────────────── Variable Renaming ───────────────────────────

_counter = [0]

def fresh(term: Term, mapping: Dict[str, Var]) -> Term:
    if isinstance(term, Var):
        if term.name not in mapping:
            _counter[0] += 1
            mapping[term.name] = Var(f"_{term.name}_{_counter[0]}")
        return mapping[term.name]
    if isinstance(term, Compound):
        return Compound(term.functor, [fresh(a, mapping) for a in term.args])
    return term


def rename_clause(head: Term, body: List[Term]) -> Tuple[Term, List[Term]]:
    mapping: Dict[str, Var] = {}
    new_head = fresh(head, mapping)
    new_body = [fresh(g, mapping) for g in body]
    return new_head, new_body


# ─────────────────────────── Inference Engine ───────────────────────────

class PrologEngine:
    def __init__(self, kb_text: str):
        self.clauses: List[Tuple] = parse_kb(kb_text)

    def query(self, goal_text: str) -> Tuple[bool, List[str]]:
        """
        Returns (success, trace).
        trace is a list of human-readable deduction steps.
        """
        tokens = tokenize(goal_text + '.')
        goal, _ = parse_term(tokens, 0)
        trace: List[str] = []
        env = next(self._solve([goal], {}, trace), None)
        if env is not None:
            result = substitute(goal, env)
            trace.append(f"✓ Query PROVED: {result}")
            return True, trace
        else:
            trace.append(f"✗ Query FAILED: {goal}")
            return False, trace

    def _solve(self, goals: List[Term], env: Env,
               trace: List[str], depth: int = 0) -> Generator[Env, None, None]:
        if not goals:
            yield env
            return

        goal = substitute(goals[0], env)
        rest = goals[1:]
        indent = "  " * depth

        # Negation-as-failure
        if isinstance(goal, Compound) and goal.functor == 'naf':
            inner_sub = substitute(goal.args[0], env)
            inner_trace: List[str] = []
            succeeded = next(self._solve([inner_sub], env, inner_trace, depth + 1), None)
            if succeeded is None:
                trace.append(f"{indent}NAF: not({inner_sub}) holds (inner goal failed)")
                yield from self._solve(rest, env, trace, depth)
            else:
                trace.append(f"{indent}NAF: not({inner_sub}) fails (inner goal succeeded)")
            return

        # Try each matching clause
        matched = False
        for head, body in self.clauses:
            new_head, new_body = rename_clause(head, body)
            new_env = unify(goal, new_head, env)
            if new_env is None:
                continue
            matched = True
            if body:
                trace.append(f"{indent}TRY: {substitute(goal, new_env)}  ←  "
                              f"{', '.join(repr(substitute(b, new_env)) for b in new_body)}")
            else:
                trace.append(f"{indent}FACT: {substitute(goal, new_env)}")
            new_goals = new_body + rest
            yield from self._solve(new_goals, new_env, trace, depth + 1)

        if not matched:
            trace.append(f"{indent}NO CLAUSE for: {goal}")


# ─────────────────────────── Load from file ───────────────────────────

def load_engine(path: str) -> PrologEngine:
    with open(path) as f:
        return PrologEngine(f.read())
