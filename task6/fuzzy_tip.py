import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip     = ctrl.Consequent(np.arange(0, 26, 1), 'tip')
quality['poor']    = fuzz.trimf(quality.universe, [0, 0, 5])
quality['average'] = fuzz.trimf(quality.universe, [0, 5, 10])
quality['good']    = fuzz.trimf(quality.universe, [5, 10, 10])
service['poor']       = fuzz.trimf(service.universe, [0, 0, 5])
service['acceptable'] = fuzz.trimf(service.universe, [0, 5, 10])
service['amazing']    = fuzz.trimf(service.universe, [5, 10, 10])
tip['low']    = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high']   = fuzz.trimf(tip.universe, [13, 25, 25])
rule1 = ctrl.Rule(quality['poor']  | service['poor'],   tip['low'])
rule2 = ctrl.Rule(service['acceptable'],                 tip['medium'])
rule3 = ctrl.Rule(service['amazing'] | quality['good'], tip['high'])
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping      = ctrl.ControlSystemSimulation(tipping_ctrl)
for q, s in [(6.5, 9.8), (0.0, 0.0), (10.0, 10.0), (5.0, 5.0)]:
    tipping.input['quality'] = q
    tipping.input['service'] = s
    tipping.compute()
    print(f"Quality={q}  Service={s}  ->  Tip: {tipping.output['tip']:.2f}%")
