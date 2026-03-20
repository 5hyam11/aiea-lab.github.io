# Fuzzy Logic with scikit-fuzzy — 1-Page Tutorial

## What is Fuzzy Logic?
Fuzzy logic extends Boolean logic by allowing truth values between 0 and 1.
Instead of TRUE/FALSE, a value like "temperature is hot" can be 0.72 (72% hot).
Invented by Lotfi Zadeh in 1965 at UC Berkeley.

## Installation
pip install scikit-fuzzy numpy matplotlib scipy networkx

## Key Concepts
- Universe of Discourse: the numeric range a variable can take
- Membership Function: maps each value to a degree in [0,1]
- Fuzzy Rule: IF-THEN statement
- Defuzzification: converts fuzzy output back to a crisp number

## Minimal Example
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
tip     = ctrl.Consequent(np.arange(0, 26, 1), 'tip')
quality['poor'] = fuzz.trimf(quality.universe, [0, 0, 5])
quality['good'] = fuzz.trimf(quality.universe, [5, 10, 10])
tip['low']  = fuzz.trimf(tip.universe, [0, 0, 13])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

## Run the Demo
python3 fuzzy_tip.py

## Expected Output
Quality=6.5  Service=9.8  ->  Tip: 19.85%
Quality=0.0  Service=0.0  ->  Tip:  4.33%
Quality=10.0 Service=10.0 ->  Tip: 21.00%
Quality=5.0  Service=5.0  ->  Tip: 12.67%

## GitHub Repo
https://github.com/5hyam11/aiea-lab.github.io/tree/shyam_auditor
