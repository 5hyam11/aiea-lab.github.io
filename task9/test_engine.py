"""
test_engine.py — Offline tests for the Prolog engine + graph nodes.
Does NOT require an OpenAI key. Tests the engine, parser, and
the solve/refine nodes in isolation.

Run:
    python test_engine.py
"""

from prolog_engine import PrologEngine

KB_LINES = [
    "planet(mercury, terrestrial, 0, no).",
    "planet(venus, terrestrial, 0, no).",
    "planet(earth, terrestrial, 1, no).",
    "planet(mars, terrestrial, 2, no).",
    "planet(jupiter, gas_giant, 95, yes).",
    "planet(saturn, gas_giant, 146, yes).",
    "planet(uranus, ice_giant, 28, yes).",
    "planet(neptune, ice_giant, 16, yes).",
    "orbits_sun(mercury, 1).", "orbits_sun(venus, 2).", "orbits_sun(earth, 3).",
    "orbits_sun(mars, 4).", "orbits_sun(jupiter, 5).", "orbits_sun(saturn, 6).",
    "orbits_sun(uranus, 7).", "orbits_sun(neptune, 8).",
    "diameter_km(mercury, 4879).", "diameter_km(venus, 12104).",
    "diameter_km(earth, 12742).", "diameter_km(mars, 6779).",
    "diameter_km(jupiter, 139820).", "diameter_km(saturn, 116460).",
    "diameter_km(uranus, 50724).", "diameter_km(neptune, 49244).",
    "inner_planet(X) :- orbits_sun(X, N), N =< 4.",
    "outer_planet(X) :- orbits_sun(X, N), N > 4.",
    "large_planet(X) :- diameter_km(X, D), D > 50000.",
    "has_rings(X) :- planet(X, _, _, yes).",
    "has_moons(X) :- planet(X, _, M, _), M > 0.",
    "neighbors(X, Y) :- orbits_sun(X, Nx), orbits_sun(Y, Ny), Ny is Nx + 1.",
    "neighbors(X, Y) :- orbits_sun(X, Nx), orbits_sun(Y, Ny), Ny is Nx - 1.",
    "habitable_candidate(X) :- inner_planet(X), has_moons(X).",
    "gas_or_ice(X) :- planet(X, gas_giant, _, _).",
    "gas_or_ice(X) :- planet(X, ice_giant, _, _).",
]


def make_engine():
    e = PrologEngine()
    for line in KB_LINES:
        e.load_line(line)
    return e


def run_tests():
    tests = [
        # (description, goal, expected_result, expected_count_min)
        ("large_planet(X) returns 3 planets",       "large_planet(X)",       True,  3),
        ("inner_planet(mars) is TRUE",              "inner_planet(mars)",    True,  1),
        ("outer_planet(mercury) is FALSE",          "outer_planet(mercury)", False, 0),
        ("has_rings(saturn) is TRUE",               "has_rings(saturn)",     True,  1),
        ("has_rings(earth) is FALSE",               "has_rings(earth)",      False, 0),
        ("has_moons(X) returns 6 planets",          "has_moons(X)",          True,  6),
        ("neighbors(earth, X) returns 2 neighbors", "neighbors(earth, X)",   True,  2),
        ("habitable_candidate(X) returns earth",    "habitable_candidate(X)",True,  1),
        ("gas_or_ice(X) returns 4 planets",         "gas_or_ice(X)",         True,  4),
        ("planet(pluto,...) is FALSE",              "planet(pluto, X, Y, Z)",False, 0),
    ]

    passed = 0
    failed = 0
    for desc, goal, expected_bool, expected_min in tests:
        e = make_engine()
        solutions, _ = e.query(goal)
        actual_bool = len(solutions) > 0
        ok = (actual_bool == expected_bool) and (len(solutions) >= expected_min)
        status = "✓ PASS" if ok else "✗ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  {desc}")
        print(f"         got {len(solutions)} solutions, expected bool={expected_bool}, min={expected_min}")

    print(f"\nResults: {passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    print("=" * 55)
    print("Task 9 — Prolog Engine Test Suite (offline, no API key)")
    print("=" * 55)
    success = run_tests()
    if not success:
        raise SystemExit(1)
