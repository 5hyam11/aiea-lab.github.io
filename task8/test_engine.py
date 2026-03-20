"""
test_engine.py
Run the Prolog engine standalone — NO OpenAI key needed.
Tests the knowledge base and prints inference traces.
"""
from prolog_engine import load_engine

KB_PATH = "knowledge_base.pl"

TEST_CASES = [
    # (query_string, expected_result)
    ("is_a(tweety, animal)",         True),
    ("is_a(sam, bird)",              True),   # penguin → bird rule
    ("has_property(sam, can_fly)",   False),  # sam has no can_fly fact
    ("is_flightless(sam)",           True),   # bird + \+ can_fly
    ("is_carnivore(rex)",            True),   # eats(rex, meat)
    ("is_carnivore(tweety)",         False),
    ("is_pet(whiskers)",             True),
    ("is_domestic(koko)",            True),
    ("is_a(nemo, animal)",           True),
    ("is_herbivore(nemo)",           True),
]

def run_tests():
    engine = load_engine(KB_PATH)
    passed = 0
    for query, expected in TEST_CASES:
        success, trace = engine.query(query)
        status = "PASS ✓" if (success == expected) else "FAIL ✗"
        if success == expected:
            passed += 1

        print(f"\n{'─'*56}")
        print(f"[{status}] {query}  →  {'TRUE' if success else 'FALSE'}  (expected {'TRUE' if expected else 'FALSE'})")
        print("Trace:")
        for step in trace:
            print(f"  {step}")

    print(f"\n{'='*56}")
    print(f"Results: {passed}/{len(TEST_CASES)} passed")

if __name__ == "__main__":
    run_tests()
