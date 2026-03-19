from pyswip import Prolog

prolog = Prolog()
prolog.consult("solar_system.pl")

queries = [
    ("Inner planets", "inner_planet(P)"),
    ("Outer planets", "outer_planet(P)"),
    ("Large planets", "large_planet(P)"),
    ("Neighbors of Earth", "neighbors(earth, P)"),
    ("Habitable candidates", "habitable_candidate(P)"),
    ("Planets with moons", "has_moons(P)"),
]

for title, query in queries:
    results = [r["P"] for r in prolog.query(query)]
    print(f"{title}: {results}")
