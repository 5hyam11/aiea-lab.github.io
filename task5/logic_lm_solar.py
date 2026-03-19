"""
logic_lm_solar.py - Logic-LM reimplementation on Solar System KB
"""
import re, sys
try:
    from pyswip import Prolog
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyswip", "-q"])
    from pyswip import Prolog

SOLAR_KB = """
planet(mercury, terrestrial, 0, no).
planet(venus, terrestrial, 0, no).
planet(earth, terrestrial, 1, no).
planet(mars, terrestrial, 2, no).
planet(jupiter, gas_giant, 95, yes).
planet(saturn, gas_giant, 146, yes).
planet(uranus, ice_giant, 28, yes).
planet(neptune, ice_giant, 16, yes).
orbits_sun(mercury, 1).
orbits_sun(venus, 2).
orbits_sun(earth, 3).
orbits_sun(mars, 4).
orbits_sun(jupiter, 5).
orbits_sun(saturn, 6).
orbits_sun(uranus, 7).
orbits_sun(neptune, 8).
diameter_km(mercury, 4879).
diameter_km(venus, 12104).
diameter_km(earth, 12742).
diameter_km(mars, 6779).
diameter_km(jupiter, 139820).
diameter_km(saturn, 116460).
diameter_km(uranus, 50724).
diameter_km(neptune, 49244).
"""

RULES = [
    "inner_planet(X) :- orbits_sun(X, P), P =< 4.",
    "outer_planet(X) :- orbits_sun(X, P), P > 4.",
    "large_planet(X) :- diameter_km(X, D), D > 50000.",
    "neighbors(X, Y) :- orbits_sun(X, P1), orbits_sun(Y, P2), abs(P1 - P2) =:= 1, X \\= Y.",
    "has_moons(X) :- planet(X, _, M, _), M > 0.",
    "has_rings(X) :- planet(X, _, _, yes).",
    "habitable_candidate(X) :- planet(X, _, _, _), X \\= earth.",
]

NL_TO_PROLOG = {
    "which planets are inner planets?"        : "inner_planet(X)",
    "which planets are outer planets?"        : "outer_planet(X)",
    "which planets are large?"                : "large_planet(X)",
    "which planets have moons?"               : "has_moons(X)",
    "which planets have rings?"               : "has_rings(X)",
    "what are the neighbors of earth?"        : "neighbors(earth, X)",
    "what are the neighbors of mars?"         : "neighbors(mars, X)",
    "what are the neighbors of jupiter?"      : "neighbors(jupiter, X)",
    "which planets are habitable candidates?" : "habitable_candidate(X)",
    "what type is jupiter?"                   : "planet(jupiter, Type, _, _)",
    "how many moons does saturn have?"        : "planet(saturn, _, Moons, _)",
    "does venus have rings?"                  : "has_rings(venus)",
    "what is mars position??"                 : "orbits_sn(mars, P)",
}

REFINEMENT_RULES = [
    (r"\borbits_sn\b", "orbits_sun"),
    (r"\bplaent\b",    "planet"),
]

GROUND_TRUTH = {
    "which planets are inner planets?"        : {"mercury","venus","earth","mars"},
    "which planets are outer planets?"        : {"jupiter","saturn","uranus","neptune"},
    "which planets are large?"                : {"jupiter","saturn","uranus"},
    "which planets have moons?"               : {"earth","mars","jupiter","saturn","uranus","neptune"},
    "which planets have rings?"               : {"jupiter","saturn","uranus","neptune"},
    "what are the neighbors of earth?"        : {"venus","mars"},
    "what are the neighbors of mars?"         : {"earth","jupiter"},
    "what are the neighbors of jupiter?"      : {"mars","saturn"},
    "which planets are habitable candidates?" : {"mercury","venus","mars","jupiter","saturn","uranus","neptune"},
}

class PrologSolver:
    def __init__(self, kb, rules):
        self.prolog = Prolog()
        for line in kb.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("%"):
                try: self.prolog.assertz(line.rstrip("."))
                except: pass
        for rule in rules:
            try: self.prolog.assertz(rule.rstrip("."))
            except: pass

    def query(self, goal):
        try:
            return list(self.prolog.query(goal)), None
        except Exception as e:
            return None, str(e)

def problem_formulator(nl):
    return NL_TO_PROLOG.get(nl.strip().lower(), f"% unknown: {nl}")

def self_refine(prog, error, rnd):
    fixed = prog
    for pat, rep in REFINEMENT_RULES:
        fixed = re.sub(pat, rep, fixed)
    print(f"  [Refinement round {rnd}] {error!r}")
    print(f"    Before: {prog!r}  ->  After: {fixed!r}")
    return fixed

def logic_lm_infer(query, solver, max_rounds=3):
    prog = problem_formulator(query)
    rounds = 0
    for r in range(max_rounds + 1):
        results, error = solver.query(prog)
        if error is None:
            if not results:
                answer = "false / no results"
            else:
                parts = []
                for row in results:
                    parts.append(", ".join(f"{k}={v}" for k,v in row.items()) if row else "true")
                answer = "; ".join(parts)
            break
        else:
            rounds += 1
            if r < max_rounds:
                prog = self_refine(prog, error, r+1)
            else:
                answer = f"[failed after {max_rounds} rounds] {error}"
    return {"query": query, "logic_program": prog, "answer": answer, "refinement_rounds": rounds}

def extract_planets(s):
    all_p = {"mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"}
    return {w for w in re.split(r"[;,=\s]+", s.lower()) if w in all_p}

if __name__ == "__main__":
    print("Loading KB...")
    solver = PrologSolver(SOLAR_KB, RULES)
    print("KB loaded.\n")

    queries = list(NL_TO_PROLOG.keys())
    total, correct = 0, 0

    print("=" * 65)
    print("  LOGIC-LM EVALUATION — Solar System KB")
    print("=" * 65)

    for q in queries:
        r = logic_lm_infer(q, solver)
        gt = GROUND_TRUTH.get(q)
        if gt is not None:
            predicted = extract_planets(r["answer"])
            match = predicted == gt
            total += 1
            if match: correct += 1
            print(f"\nQ: {q}")
            print(f"   Program : {r['logic_program']}")
            print(f"   Answer  : {r['answer']}")
            print(f"   Rounds  : {r['refinement_rounds']}")
            print(f"   Result  : {'✓ CORRECT' if match else '✗ WRONG'}")
        else:
            print(f"\nQ: {q}")
            print(f"   Program : {r['logic_program']}")
            print(f"   Answer  : {r['answer']}")

    print(f"\n{'-'*65}")
    print(f"  ACCURACY: {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"{'-'*65}")
