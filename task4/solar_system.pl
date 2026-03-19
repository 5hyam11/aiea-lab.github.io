% Solar System Knowledge Base

% Facts: planet(Name, Type, Moons, HasRings)
planet(mercury, terrestrial, 0, no).
planet(venus,   terrestrial, 0, no).
planet(earth,   terrestrial, 1, no).
planet(mars,    terrestrial, 2, no).
planet(jupiter, gas_giant,   95, yes).
planet(saturn,  gas_giant,   146, yes).
planet(uranus,  ice_giant,   28, yes).
planet(neptune, ice_giant,   16, yes).

% Facts: orbits_sun(Planet, Position)
orbits_sun(mercury, 1).
orbits_sun(venus,   2).
orbits_sun(earth,   3).
orbits_sun(mars,    4).
orbits_sun(jupiter, 5).
orbits_sun(saturn,  6).
orbits_sun(uranus,  7).
orbits_sun(neptune, 8).

% Facts: diameter_km(Planet, Diameter)
diameter_km(mercury, 4879).
diameter_km(venus,   12104).
diameter_km(earth,   12742).
diameter_km(mars,    6779).
diameter_km(jupiter, 139820).
diameter_km(saturn,  116460).
diameter_km(uranus,  50724).
diameter_km(neptune, 49244).

% Fact: has_life
has_life(earth).

% Rules
inner_planet(P) :- orbits_sun(P, Pos), Pos =< 4.
outer_planet(P) :- orbits_sun(P, Pos), Pos > 4.
large_planet(P) :- diameter_km(P, D), D > 50000.
has_moons(P) :- planet(P, _, Moons, _), Moons > 0.
habitable_candidate(P) :- planet(P, terrestrial, _, _), \+ has_life(P).
neighbors(P1, P2) :- orbits_sun(P1, A), orbits_sun(P2, B), P1 \= P2, abs(A-B) =:= 1.
