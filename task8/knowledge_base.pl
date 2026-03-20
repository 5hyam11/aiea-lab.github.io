% ============================================================
% Animal Kingdom Knowledge Base
% A simple Prolog KB with ~15 facts and ~8 rules
% ============================================================

% --- FACTS: is_a(Individual, Category) ---
is_a(tweety, bird).
is_a(sam, penguin).
is_a(rex, dog).
is_a(whiskers, cat).
is_a(nemo, fish).
is_a(goldie, fish).
is_a(leo, lion).
is_a(bella, dog).
is_a(koko, parrot).

% --- FACTS: has_property ---
has_property(tweety, can_fly).
has_property(koko, can_fly).
has_property(rex, has_fur).
has_property(whiskers, has_fur).
has_property(leo, has_fur).
has_property(bella, has_fur).
has_property(nemo, lives_in_water).
has_property(goldie, lives_in_water).
has_property(sam, lives_in_water).

% --- FACTS: eats ---
eats(rex, meat).
eats(leo, meat).
eats(whiskers, fish).
eats(tweety, seeds).
eats(koko, seeds).
eats(nemo, plants).
eats(sam, fish).

% --- RULES ---

% A penguin is a bird
is_a(X, bird) :- is_a(X, penguin).

% Birds are animals
is_a(X, animal) :- is_a(X, bird).

% Dogs, cats, fish, and lions are animals
is_a(X, animal) :- is_a(X, dog).
is_a(X, animal) :- is_a(X, cat).
is_a(X, animal) :- is_a(X, fish).
is_a(X, animal) :- is_a(X, lion).
is_a(X, animal) :- is_a(X, parrot).

% Something is a pet if it's a dog or cat
is_pet(X) :- is_a(X, dog).
is_pet(X) :- is_a(X, cat).

% Something is a carnivore if it eats meat or fish
is_carnivore(X) :- eats(X, meat).
is_carnivore(X) :- eats(X, fish).

% Something is a herbivore if it eats plants or seeds
is_herbivore(X) :- eats(X, plants).
is_herbivore(X) :- eats(X, seeds).

% Something is domestic if it is a pet or a parrot
is_domestic(X) :- is_pet(X).
is_domestic(X) :- is_a(X, parrot).

% A bird that cannot fly explicitly is flightless
is_flightless(X) :- is_a(X, bird), \+ has_property(X, can_fly).
