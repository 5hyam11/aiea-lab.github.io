"""
Backward Chaining System for First Order Logic (FOL)
CSE 240 - Task 7
"""

from typing import Optional


class Term:
    def __init__(self, name: str, args: list = None):
        self.name = name
        self.args = args or []

    def is_variable(self) -> bool:
        return self.name[0].isupper() and not self.args

    def __repr__(self):
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(repr, self.args))})"

    def __eq__(self, other):
        return isinstance(other, Term) and self.name == other.name and self.args == other.args

    def __hash__(self):
        return hash((self.name, tuple(self.args)))


class Literal:
    def __init__(self, predicate: str, args: list, negated: bool = False):
        self.predicate = predicate
        self.args = args
        self.negated = negated

    def __repr__(self):
        base = f"{self.predicate}({', '.join(map(repr, self.args))})"
        return f"NOT {base}" if self.negated else base

    def __eq__(self, other):
        return (isinstance(other, Literal) and
                self.predicate == other.predicate and
                self.args == other.args and
                self.negated == other.negated)


class Rule:
    def __init__(self, head: Literal, body: list):
        self.head = head
        self.body = body

    def __repr__(self):
        if not self.body:
            return repr(self.head)
        return f"{self.head} :- {', '.join(map(repr, self.body))}"


def unify(x, y, subst: dict) -> Optional[dict]:
    if subst is None:
        return None
    if x == y:
        return subst
    if isinstance(x, Term) and x.is_variable():
        return unify_var(x, y, subst)
    if isinstance(y, Term) and y.is_variable():
        return unify_var(y, x, subst)
    if isinstance(x, Term) and isinstance(y, Term):
        if x.name != y.name or len(x.args) != len(y.args):
            return None
        for xi, yi in zip(x.args, y.args):
            subst = unify(xi, yi, subst)
            if subst is None:
                return None
        return subst
    if isinstance(x, Literal) and isinstance(y, Literal):
        if x.predicate != y.predicate or x.negated != y.negated or len(x.args) != len(y.args):
            return None
        for xi, yi in zip(x.args, y.args):
            subst = unify(xi, yi, subst)
            if subst is None:
                return None
        return subst
    return None


def unify_var(var: Term, x, subst: dict) -> Optional[dict]:
    if var.name in subst:
        return unify(subst[var.name], x, subst)
    if isinstance(x, Term) and x.is_variable() and x.name in subst:
        return unify(var, subst[x.name], subst)
    if occurs_check(var, x, subst):
        return None
    return {**subst, var.name: x}


def occurs_check(var: Term, x, subst: dict) -> bool:
    if var == x:
        return True
    if isinstance(x, Term) and x.is_variable() and x.name in subst:
        return occurs_check(var, subst[x.name], subst)
    if isinstance(x, Term) and x.args:
        return any(occurs_check(var, arg, subst) for arg in x.args)
    return False


def apply_subst(term, subst: dict):
    if isinstance(term, Term):
        if term.is_variable():
            if term.name in subst:
                return apply_subst(subst[term.name], subst)
            return term
        return Term(term.name, [apply_subst(a, subst) for a in term.args])
    if isinstance(term, Literal):
        return Literal(term.predicate,
                       [apply_subst(a, subst) for a in term.args],
                       term.negated)
    return term


_counter = [0]

def fresh_rule(rule: Rule) -> Rule:
    _counter[0] += 1
    suffix = f"_{_counter[0]}"

    def rename_term(t):
        if isinstance(t, Term):
            if t.is_variable():
                return Term(t.name + suffix)
            return Term(t.name, [rename_term(a) for a in t.args])
        return t

    def rename_lit(lit):
        return Literal(lit.predicate, [rename_term(a) for a in lit.args], lit.negated)

    return Rule(rename_lit(rule.head), [rename_lit(b) for b in rule.body])


class KnowledgeBase:
    def __init__(self):
        self.facts = []
        self.rules = []

    def tell_fact(self, predicate: str, *args):
        self.facts.append(Literal(predicate, [Term(a) for a in args]))

    def tell_rule(self, head: Literal, body: list):
        self.rules.append(Rule(head, body))

    def ask(self, query: Literal, verbose: bool = False):
        results = []
        for subst in self._bc_or(query, {}, depth=0, verbose=verbose):
            results.append(subst)
        return results

    def _bc_or(self, goal: Literal, subst: dict, depth: int, verbose: bool):
        indent = "  " * depth
        if verbose:
            print(f"{indent}? {apply_subst(goal, subst)}")
        for fact in self.facts:
            s = unify(goal, fact, subst)
            if s is not None:
                if verbose:
                    print(f"{indent}  matched fact: {fact}")
                yield s
        for rule in self.rules:
            fresh = fresh_rule(rule)
            s = unify(goal, fresh.head, subst)
            if s is not None:
                if verbose:
                    print(f"{indent}  applying rule: {fresh}")
                yield from self._bc_and(fresh.body, s, depth + 1, verbose)

    def _bc_and(self, goals: list, subst: dict, depth: int, verbose: bool):
        if not goals:
            yield subst
            return
        first, *rest = goals
        for s in self._bc_or(first, subst, depth, verbose):
            yield from self._bc_and(rest, s, depth, verbose)


def build_family_kb() -> KnowledgeBase:
    kb = KnowledgeBase()

    for parent, child in [("tom","bob"),("tom","liz"),("bob","ann"),
                           ("bob","pat"),("pat","jim"),("pat","sue"),
                           ("mary","bob"),("mary","liz")]:
        kb.tell_fact("parent", parent, child)

    for m in ["tom","bob","pat","jim"]:
        kb.tell_fact("male", m)
    for f in ["liz","ann","sue","mary"]:
        kb.tell_fact("female", f)

    for person, age in [("tom","70"),("bob","45"),("liz","43"),("mary","68"),
                        ("ann","20"),("pat","22"),("jim","5"),("sue","3")]:
        kb.tell_fact("age", person, age)

    kb.tell_rule(
        Literal("grandparent", [Term("X"), Term("Z")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("parent", [Term("Y"), Term("Z")])])

    kb.tell_rule(
        Literal("father", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("male", [Term("X")])])

    kb.tell_rule(
        Literal("mother", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("female", [Term("X")])])

    kb.tell_rule(
        Literal("sibling", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("P"), Term("X")]),
         Literal("parent", [Term("P"), Term("Y")])])

    kb.tell_rule(
        Literal("ancestor", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("X"), Term("Y")])])

    kb.tell_rule(
        Literal("ancestor", [Term("X"), Term("Z")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("ancestor", [Term("Y"), Term("Z")])])

    return kb


def run_tests():
    kb = build_family_kb()

    def show(label, q, results):
        print(f"\n{'─'*50}")
        print(f"Query: {q}")
        print(f"Results ({len(results)} solutions):")
        if not results:
            print("  (none)")
        for s in results:
            args = [str(apply_subst(a, s)) for a in q.args]
            print(f"  {q.predicate}({', '.join(args)})")

    print("=" * 50)
    print("  BACKWARD CHAINING — FOL TEST SUITE")
    print("=" * 50)

    show("parent", Literal("parent", [Term("X"), Term("Y")]), kb.ask(Literal("parent", [Term("X"), Term("Y")])))
    show("grandparent", Literal("grandparent", [Term("X"), Term("Y")]), kb.ask(Literal("grandparent", [Term("X"), Term("Y")])))

    q3 = Literal("father", [Term("tom"), Term("bob")])
    r3 = kb.ask(q3)
    print(f"\n{'─'*50}")
    print(f"Query: {q3}")
    print(f"Result: {'TRUE ✓' if r3 else 'FALSE ✗'}")

    show("ancestor(X, jim)", Literal("ancestor", [Term("X"), Term("jim")]), kb.ask(Literal("ancestor", [Term("X"), Term("jim")])))
    show("sibling(bob, Y)", Literal("sibling", [Term("bob"), Term("Y")]), kb.ask(Literal("sibling", [Term("bob"), Term("Y")])))
    show("mother", Literal("mother", [Term("X"), Term("Y")]), kb.ask(Literal("mother", [Term("X"), Term("Y")])))

    print(f"\n{'─'*50}")
    print("Verbose trace: grandparent(tom, Z)?")
    kb.ask(Literal("grandparent", [Term("tom"), Term("Z")]), verbose=True)

    q8 = Literal("parent", [Term("ann"), Term("Y")])
    r8 = kb.ask(q8)
    print(f"\n{'─'*50}")
    print(f"Query: {q8}")
    print(f"Result: {'TRUE' if r8 else 'FALSE ✗ (correct — ann has no children)'}")

    print(f"\n{'='*50}")
    print("All tests complete.")


if __name__ == "__main__":
    run_tests()
EOFcat > task7/backward_chaining.py << 'EOF'
"""
Backward Chaining System for First Order Logic (FOL)
CSE 240 - Task 7
"""

from typing import Optional


class Term:
    def __init__(self, name: str, args: list = None):
        self.name = name
        self.args = args or []

    def is_variable(self) -> bool:
        return self.name[0].isupper() and not self.args

    def __repr__(self):
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(repr, self.args))})"

    def __eq__(self, other):
        return isinstance(other, Term) and self.name == other.name and self.args == other.args

    def __hash__(self):
        return hash((self.name, tuple(self.args)))


class Literal:
    def __init__(self, predicate: str, args: list, negated: bool = False):
        self.predicate = predicate
        self.args = args
        self.negated = negated

    def __repr__(self):
        base = f"{self.predicate}({', '.join(map(repr, self.args))})"
        return f"NOT {base}" if self.negated else base

    def __eq__(self, other):
        return (isinstance(other, Literal) and
                self.predicate == other.predicate and
                self.args == other.args and
                self.negated == other.negated)


class Rule:
    def __init__(self, head: Literal, body: list):
        self.head = head
        self.body = body

    def __repr__(self):
        if not self.body:
            return repr(self.head)
        return f"{self.head} :- {', '.join(map(repr, self.body))}"


def unify(x, y, subst: dict) -> Optional[dict]:
    if subst is None:
        return None
    if x == y:
        return subst
    if isinstance(x, Term) and x.is_variable():
        return unify_var(x, y, subst)
    if isinstance(y, Term) and y.is_variable():
        return unify_var(y, x, subst)
    if isinstance(x, Term) and isinstance(y, Term):
        if x.name != y.name or len(x.args) != len(y.args):
            return None
        for xi, yi in zip(x.args, y.args):
            subst = unify(xi, yi, subst)
            if subst is None:
                return None
        return subst
    if isinstance(x, Literal) and isinstance(y, Literal):
        if x.predicate != y.predicate or x.negated != y.negated or len(x.args) != len(y.args):
            return None
        for xi, yi in zip(x.args, y.args):
            subst = unify(xi, yi, subst)
            if subst is None:
                return None
        return subst
    return None


def unify_var(var: Term, x, subst: dict) -> Optional[dict]:
    if var.name in subst:
        return unify(subst[var.name], x, subst)
    if isinstance(x, Term) and x.is_variable() and x.name in subst:
        return unify(var, subst[x.name], subst)
    if occurs_check(var, x, subst):
        return None
    return {**subst, var.name: x}


def occurs_check(var: Term, x, subst: dict) -> bool:
    if var == x:
        return True
    if isinstance(x, Term) and x.is_variable() and x.name in subst:
        return occurs_check(var, subst[x.name], subst)
    if isinstance(x, Term) and x.args:
        return any(occurs_check(var, arg, subst) for arg in x.args)
    return False


def apply_subst(term, subst: dict):
    if isinstance(term, Term):
        if term.is_variable():
            if term.name in subst:
                return apply_subst(subst[term.name], subst)
            return term
        return Term(term.name, [apply_subst(a, subst) for a in term.args])
    if isinstance(term, Literal):
        return Literal(term.predicate,
                       [apply_subst(a, subst) for a in term.args],
                       term.negated)
    return term


_counter = [0]

def fresh_rule(rule: Rule) -> Rule:
    _counter[0] += 1
    suffix = f"_{_counter[0]}"

    def rename_term(t):
        if isinstance(t, Term):
            if t.is_variable():
                return Term(t.name + suffix)
            return Term(t.name, [rename_term(a) for a in t.args])
        return t

    def rename_lit(lit):
        return Literal(lit.predicate, [rename_term(a) for a in lit.args], lit.negated)

    return Rule(rename_lit(rule.head), [rename_lit(b) for b in rule.body])


class KnowledgeBase:
    def __init__(self):
        self.facts = []
        self.rules = []

    def tell_fact(self, predicate: str, *args):
        self.facts.append(Literal(predicate, [Term(a) for a in args]))

    def tell_rule(self, head: Literal, body: list):
        self.rules.append(Rule(head, body))

    def ask(self, query: Literal, verbose: bool = False):
        results = []
        for subst in self._bc_or(query, {}, depth=0, verbose=verbose):
            results.append(subst)
        return results

    def _bc_or(self, goal: Literal, subst: dict, depth: int, verbose: bool):
        indent = "  " * depth
        if verbose:
            print(f"{indent}? {apply_subst(goal, subst)}")
        for fact in self.facts:
            s = unify(goal, fact, subst)
            if s is not None:
                if verbose:
                    print(f"{indent}  matched fact: {fact}")
                yield s
        for rule in self.rules:
            fresh = fresh_rule(rule)
            s = unify(goal, fresh.head, subst)
            if s is not None:
                if verbose:
                    print(f"{indent}  applying rule: {fresh}")
                yield from self._bc_and(fresh.body, s, depth + 1, verbose)

    def _bc_and(self, goals: list, subst: dict, depth: int, verbose: bool):
        if not goals:
            yield subst
            return
        first, *rest = goals
        for s in self._bc_or(first, subst, depth, verbose):
            yield from self._bc_and(rest, s, depth, verbose)


def build_family_kb() -> KnowledgeBase:
    kb = KnowledgeBase()

    for parent, child in [("tom","bob"),("tom","liz"),("bob","ann"),
                           ("bob","pat"),("pat","jim"),("pat","sue"),
                           ("mary","bob"),("mary","liz")]:
        kb.tell_fact("parent", parent, child)

    for m in ["tom","bob","pat","jim"]:
        kb.tell_fact("male", m)
    for f in ["liz","ann","sue","mary"]:
        kb.tell_fact("female", f)

    for person, age in [("tom","70"),("bob","45"),("liz","43"),("mary","68"),
                        ("ann","20"),("pat","22"),("jim","5"),("sue","3")]:
        kb.tell_fact("age", person, age)

    kb.tell_rule(
        Literal("grandparent", [Term("X"), Term("Z")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("parent", [Term("Y"), Term("Z")])])

    kb.tell_rule(
        Literal("father", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("male", [Term("X")])])

    kb.tell_rule(
        Literal("mother", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("female", [Term("X")])])

    kb.tell_rule(
        Literal("sibling", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("P"), Term("X")]),
         Literal("parent", [Term("P"), Term("Y")])])

    kb.tell_rule(
        Literal("ancestor", [Term("X"), Term("Y")]),
        [Literal("parent", [Term("X"), Term("Y")])])

    kb.tell_rule(
        Literal("ancestor", [Term("X"), Term("Z")]),
        [Literal("parent", [Term("X"), Term("Y")]),
         Literal("ancestor", [Term("Y"), Term("Z")])])

    return kb


def run_tests():
    kb = build_family_kb()

    def show(label, q, results):
        print(f"\n{'─'*50}")
        print(f"Query: {q}")
        print(f"Results ({len(results)} solutions):")
        if not results:
            print("  (none)")
        for s in results:
            args = [str(apply_subst(a, s)) for a in q.args]
            print(f"  {q.predicate}({', '.join(args)})")

    print("=" * 50)
    print("  BACKWARD CHAINING — FOL TEST SUITE")
    print("=" * 50)

    show("parent", Literal("parent", [Term("X"), Term("Y")]), kb.ask(Literal("parent", [Term("X"), Term("Y")])))
    show("grandparent", Literal("grandparent", [Term("X"), Term("Y")]), kb.ask(Literal("grandparent", [Term("X"), Term("Y")])))

    q3 = Literal("father", [Term("tom"), Term("bob")])
    r3 = kb.ask(q3)
    print(f"\n{'─'*50}")
    print(f"Query: {q3}")
    print(f"Result: {'TRUE ✓' if r3 else 'FALSE ✗'}")

    show("ancestor(X, jim)", Literal("ancestor", [Term("X"), Term("jim")]), kb.ask(Literal("ancestor", [Term("X"), Term("jim")])))
    show("sibling(bob, Y)", Literal("sibling", [Term("bob"), Term("Y")]), kb.ask(Literal("sibling", [Term("bob"), Term("Y")])))
    show("mother", Literal("mother", [Term("X"), Term("Y")]), kb.ask(Literal("mother", [Term("X"), Term("Y")])))

    print(f"\n{'─'*50}")
    print("Verbose trace: grandparent(tom, Z)?")
    kb.ask(Literal("grandparent", [Term("tom"), Term("Z")]), verbose=True)

    q8 = Literal("parent", [Term("ann"), Term("Y")])
    r8 = kb.ask(q8)
    print(f"\n{'─'*50}")
    print(f"Query: {q8}")
    print(f"Result: {'TRUE' if r8 else 'FALSE ✗ (correct — ann has no children)'}")

    print(f"\n{'='*50}")
    print("All tests complete.")


if __name__ == "__main__":
    run_tests()
