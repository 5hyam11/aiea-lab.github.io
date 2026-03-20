"""
prolog_engine.py — Pure-Python backward-chaining Prolog engine.
Reused from Task 8 (LangChain). No external dependencies.
"""

import re
from copy import deepcopy
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Term / Literal / Rule
# ─────────────────────────────────────────────────────────────────────────────

class Term:
    def __init__(self, name: str, args: list = None):
        self.name = name
        self.args = args or []

    def is_var(self):
        return self.name[0].isupper() or self.name.startswith("_")

    def is_const(self):
        return not self.is_var() and not self.args

    def __repr__(self):
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(str(a) for a in self.args)})"

    def __eq__(self, other):
        return isinstance(other, Term) and self.name == other.name and self.args == other.args


class Literal:
    def __init__(self, pred: str, args: list):
        self.pred = pred
        self.args = args

    def __repr__(self):
        if not self.args:
            return self.pred
        return f"{self.pred}({', '.join(str(a) for a in self.args)})"


class Rule:
    def __init__(self, head: Literal, body: list):
        self.head = head
        self.body = body  # list of Literal

    def __repr__(self):
        if not self.body:
            return f"{self.head}."
        return f"{self.head} :- {', '.join(str(b) for b in self.body)}."


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(s: str):
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?|[(),+\-*/]|=:=|=\\=|=<|>=|:-|\\+|[=<>]', s)


_anon_counter = [0]

def parse_term(tokens: list, pos: int):
    """Parse a single term, including compound arithmetic like Nx + 1."""
    t = tokens[pos]
    if t == "_":
        _anon_counter[0] += 1
        t = f"_Anon{_anon_counter[0]}"
    pos += 1
    if pos < len(tokens) and tokens[pos] == "(":
        pos += 1  # consume '('
        args = []
        while tokens[pos] != ")":
            arg, pos = parse_term(tokens, pos)
            args.append(arg)
            if pos < len(tokens) and tokens[pos] == ",":
                pos += 1
        pos += 1  # consume ')'
        base = Term(t, args)
    else:
        base = Term(t)

    # Handle inline binary operators: +, -, *, /
    if pos < len(tokens) and tokens[pos] in ("+", "-", "*", "/"):
        op = tokens[pos]
        pos += 1
        rhs, pos = parse_term(tokens, pos)
        return Term(op, [base, rhs]), pos

    return base, pos


def parse_literal(tokens: list, pos: int):
    neg = False
    if pos < len(tokens) and tokens[pos] == "\\+" :
        neg = True
        pos += 1
    name = tokens[pos]
    pos += 1
    args = []
    if pos < len(tokens) and tokens[pos] == "(":
        pos += 1
        while tokens[pos] != ")":
            arg, pos = parse_term(tokens, pos)
            args.append(arg)
            if pos < len(tokens) and tokens[pos] == ",":
                pos += 1
        pos += 1
    # handle inline arithmetic "X is Expr" and comparison "X =< Y"
    if pos < len(tokens) and tokens[pos] in ("is", "=<", ">=", "<", ">", "=:=", "=\\="):
        op = tokens[pos]
        pos += 1
        rhs, pos = parse_term(tokens, pos)
        lit = Literal("__arith__", [Term(name), Term(op), rhs])
        if neg:
            lit = Literal("__neg__", [lit])
        return lit, pos
    lit = Literal(name, args)
    if neg:
        lit = Literal("__neg__", [lit])
    return lit, pos


def parse_rule(line: str) -> Optional[Rule]:
    line = line.strip().rstrip(".")
    if not line or line.startswith("%"):
        return None
    tokens = tokenize(line)
    if not tokens:
        return None
    try:
        head, pos = parse_literal(tokens, 0)
        if pos < len(tokens) and tokens[pos] == ":-":
            pos += 1
            body = []
            while pos < len(tokens):
                lit, pos = parse_literal(tokens, pos)
                body.append(lit)
                if pos < len(tokens) and tokens[pos] == ",":
                    pos += 1
            return Rule(head, body)
        return Rule(head, [])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Unification
# ─────────────────────────────────────────────────────────────────────────────

def apply_subst(term: Term, subst: dict) -> Term:
    if term.is_var():
        if term.name in subst:
            return apply_subst(subst[term.name], subst)
        return term
    if term.args:
        return Term(term.name, [apply_subst(a, subst) for a in term.args])
    return term


def unify(t1: Term, t2: Term, subst: dict) -> Optional[dict]:
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)
    if t1 == t2:
        return subst
    if t1.is_var():
        s = dict(subst)
        s[t1.name] = t2
        return s
    if t2.is_var():
        s = dict(subst)
        s[t2.name] = t1
        return s
    if t1.name == t2.name and len(t1.args) == len(t2.args):
        s = subst
        for a, b in zip(t1.args, t2.args):
            s = unify(a, b, s)
            if s is None:
                return None
        return s
    return None


def unify_literals(l1: Literal, l2: Literal, subst: dict) -> Optional[dict]:
    if l1.pred != l2.pred or len(l1.args) != len(l2.args):
        return None
    s = subst
    for a, b in zip(l1.args, l2.args):
        s = unify(a, b, s)
        if s is None:
            return None
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Rename variables in a rule
# ─────────────────────────────────────────────────────────────────────────────

_counter = [0]

def rename_rule(rule: Rule) -> Rule:
    _counter[0] += 1
    tag = f"_{_counter[0]}"

    def rename_term(t: Term) -> Term:
        if t.is_var():
            return Term(t.name + tag)
        return Term(t.name, [rename_term(a) for a in t.args])

    def rename_lit(lit: Literal) -> Literal:
        return Literal(lit.pred, [rename_term(a) for a in lit.args])

    head = rename_lit(rule.head)
    body = [rename_lit(b) for b in rule.body]
    return Rule(head, body)


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic evaluator
# ─────────────────────────────────────────────────────────────────────────────

def eval_term(term: Term, subst: dict) -> float:
    t = apply_subst(term, subst)
    if t.args:
        if t.name in ("+", "-", "*", "/"):
            l = eval_term(t.args[0], subst)
            r = eval_term(t.args[1], subst)
            return {"+": l+r, "-": l-r, "*": l*r, "/": l/r}[t.name]
    return float(t.name)


# ─────────────────────────────────────────────────────────────────────────────
# PrologEngine
# ─────────────────────────────────────────────────────────────────────────────

class PrologEngine:
    def __init__(self):
        self.rules: list[Rule] = []
        self.trace: list[str] = []

    def load_line(self, line: str):
        r = parse_rule(line)
        if r:
            self.rules.append(r)

    def query(self, goal_str: str):
        self.trace = []
        tokens = tokenize(goal_str)
        goal, _ = parse_literal(tokens, 0)
        solutions = list(self._bc_or(goal, {}))
        return solutions, self.trace

    def _bc_or(self, goal: Literal, subst: dict):
        self.trace.append(f"? {goal}")

        # built-in: arithmetic / comparison
        if goal.pred == "__arith__":
            lhs_term, op_term, rhs_term = goal.args[0], goal.args[1], goal.args[2]
            op = op_term.name
            try:
                lhs_val = eval_term(lhs_term, subst)
                rhs_val = eval_term(rhs_term, subst)
            except Exception:
                return
            if op == "is":
                s = unify(lhs_term, Term(str(int(rhs_val)) if rhs_val == int(rhs_val) else str(rhs_val)), subst)
                if s is not None:
                    self.trace.append(f"  ✓ arith {lhs_term} = {rhs_val}")
                    yield s
            else:
                result = {
                    "=<": lhs_val <= rhs_val, ">=": lhs_val >= rhs_val,
                    "<":  lhs_val <  rhs_val, ">":  lhs_val >  rhs_val,
                    "=:=": lhs_val == rhs_val, "=\\=": lhs_val != rhs_val,
                }[op]
                if result:
                    self.trace.append(f"  ✓ {lhs_val} {op} {rhs_val}")
                    yield subst
            return

        # built-in: negation-as-failure
        if goal.pred == "__neg__":
            inner = goal.args[0]
            found = next(self._bc_or(inner, subst), None)
            if found is None:
                self.trace.append(f"  ✓ \\+ {inner}")
                yield subst
            return

        for rule in self.rules:
            renamed = rename_rule(rule)
            s = unify_literals(goal, renamed.head, subst)
            if s is not None:
                self.trace.append(f"  via {renamed.head}")
                yield from self._bc_and(renamed.body, s)

    def _bc_and(self, goals: list, subst: dict):
        if not goals:
            yield subst
            return
        head, *rest = goals
        for s in self._bc_or(head, subst):
            yield from self._bc_and(rest, s)
