#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:59:32 2017

@author: jpoeppel
"""

from primo2.networks import DecisionNetwork, DynamicDecisionNetwork, d0_net, Two_TDN
from primo2.nodes import DiscreteNode, DecisionNode, UtilityNode

from primo2.inference.decision import VariableElimination
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import pandas as pd

print ("")
print ("PHD example 7.3 from Bayesian Reasoning and Machine Learning - Barber")
print ("")

net = DecisionNetwork()

education = DecisionNode("education", decisions=["do Phd", "no Phd"])  # E

income = DiscreteNode("income", values=["low", "average", "high"])  # I
nobel = DiscreteNode("nobel", values=["prize", "no prize"])  # P

costs = UtilityNode("costs")  # UC
gains = UtilityNode("gains")  # UB

# Add nodes to network. They can be treated the same
net.add_node(education)
net.add_node(income)
net.add_node(nobel)

net.add_node(costs)
net.add_node(gains)

# Add edges. Edges can either be acutal dependencies or information links.
# The type is figured out by the nodes themsevles
net.add_edge(education, costs)
net.add_edge(education, nobel)
net.add_edge(education, income)

net.add_edge(nobel, income)
net.add_edge(income, gains)

# Define CPTs: (Needs to be done AFTER the structure is defined as that)
# determines the table structure for the different nodes

income.set_probability("low", 0.1, parentValues={"education": "do Phd", "nobel": "no prize"})
income.set_probability("low", 0.2, parentValues={"education": "no Phd", "nobel": "no prize"})
income.set_probability("low", 0.01, parentValues={"education": "do Phd", "nobel": "prize"})
income.set_probability("low", 0.01, parentValues={"education": "no Phd", "nobel": "prize"})

income.set_probability("average", 0.5, parentValues={"education": "do Phd", "nobel": "no prize"})
income.set_probability("average", 0.6, parentValues={"education": "no Phd", "nobel": "no prize"})
income.set_probability("average", 0.04, parentValues={"education": "do Phd", "nobel": "prize"})
income.set_probability("average", 0.04, parentValues={"education": "no Phd", "nobel": "prize"})

income.set_probability("high", 0.4, parentValues={"education": "do Phd", "nobel": "no prize"})
income.set_probability("high", 0.2, parentValues={"education": "no Phd", "nobel": "no prize"})
income.set_probability("high", 0.95, parentValues={"education": "do Phd", "nobel": "prize"})
income.set_probability("high", 0.95, parentValues={"education": "no Phd", "nobel": "prize"})

nobel.set_probability("prize", 0.0000001, parentValues={"education": "no Phd"})
nobel.set_probability("prize", 0.001, parentValues={"education": "do Phd"})

nobel.set_probability("no prize", 0.9999999, parentValues={"education": "no Phd"})
nobel.set_probability("no prize", 0.999, parentValues={"education": "do Phd"})

# Define utilities

costs.set_utility(-50000, parentValues={"education": "do Phd"})
costs.set_utility(0, parentValues={"education": "no Phd"})

gains.set_utility(100000, parentValues={"income": "low"})
gains.set_utility(200000, parentValues={"income": "average"})
gains.set_utility(500000, parentValues={"income": "high"})

net.set_partial_ordering([education, [income, nobel]])
ve = VariableElimination(net)

print("Expected Utility for doing a Phd: {}".format(ve.expected_utility(decisions={"education": "do Phd"})))
print("Expected Utility for not doing a Phd: {}".format(ve.expected_utility(decisions={"education": "no Phd"})))
print "Optimal decision using max_sum: ", ve.max_sum("education")
print "Get optimal decision using classic algorithm: ", ve.get_optimal_decisions(["education"])

print ("")
print ("PHD example 7.4 from Bayesian Reasoning and Machine Learning - Barber")
print ("")

net = DecisionNetwork()

education = DecisionNode("education", decisions=["do Phd", "no Phd"])  # E
startup = DecisionNode("startup", decisions=["start up", "no start up"])  # S

income = DiscreteNode("income", values=["low", "average", "high"])  # I
nobel = DiscreteNode("nobel", values=["prize", "no prize"])  # P

costsEducation = UtilityNode("costsE")  # UC
costsStartUp = UtilityNode("costsS")  # US
gains = UtilityNode("gains")  # UB

# Add nodes to network. They can be treated the same
net.add_node(education)
net.add_node(startup)
net.add_node(income)
net.add_node(nobel)

net.add_node(costsEducation)
net.add_node(costsStartUp)
net.add_node(gains)

# Add edges. Edges can either be acutal dependencies or information links.
# The type is figured out by the nodes themsevles
net.add_edge(education, costsEducation)
net.add_edge(education, nobel)

net.add_edge(startup, income)
net.add_edge(startup, costsStartUp)

net.add_edge(nobel, income)
net.add_edge(income, gains)

# Define CPTs: (Needs to be done AFTER the structure is defined as that)
# determines the table structure for the different nodes

income.set_probability("low", 0.1, parentValues={"startup": "start up", "nobel": "no prize"})
income.set_probability("low", 0.2, parentValues={"startup": "no start up", "nobel": "no prize"})
income.set_probability("low", 0.005, parentValues={"startup": "start up", "nobel": "prize"})
income.set_probability("low", 0.05, parentValues={"startup": "no start up", "nobel": "prize"})

income.set_probability("average", 0.5, parentValues={"startup": "start up", "nobel": "no prize"})
income.set_probability("average", 0.6, parentValues={"startup": "no start up", "nobel": "no prize"})
income.set_probability("average", 0.005, parentValues={"startup": "start up", "nobel": "prize"})
income.set_probability("average", 0.15, parentValues={"startup": "no start up", "nobel": "prize"})

income.set_probability("high", 0.4, parentValues={"startup": "start up", "nobel": "no prize"})
income.set_probability("high", 0.2, parentValues={"startup": "no start up", "nobel": "no prize"})
income.set_probability("high", 0.99, parentValues={"startup": "start up", "nobel": "prize"})
income.set_probability("high", 0.8, parentValues={"startup": "no start up", "nobel": "prize"})

nobel.set_probability("prize", 0.0000001, parentValues={"education": "no Phd"})
nobel.set_probability("prize", 0.001, parentValues={"education": "do Phd"})

nobel.set_probability("no prize", 0.9999999, parentValues={"education": "no Phd"})
nobel.set_probability("no prize", 0.999, parentValues={"education": "do Phd"})

# Define utilities

costsEducation.set_utility(-50000, parentValues={"education": "do Phd"})
costsEducation.set_utility(0, parentValues={"education": "no Phd"})

costsStartUp.set_utility(-200000, parentValues={"startup": "start up"})
costsStartUp.set_utility(0, parentValues={"startup": "no start up"})

gains.set_utility(100000, parentValues={"income": "low"})
gains.set_utility(200000, parentValues={"income": "average"})
gains.set_utility(500000, parentValues={"income": "high"})

net.set_partial_ordering([education, nobel, startup, income])
ve = VariableElimination(net)

print("Expected Utility for doing a Phd + startup: {}".format(
    ve.expected_utility(decisions={"education": "do Phd", "startup": "start up"})))
print("Expected Utility for doing a Phd + no startup: {}".format(
    ve.expected_utility(decisions={"education": "do Phd", "startup": "no start up"})))
print("Expected Utility for not doing a Phd + startup: {}".format(
    ve.expected_utility(decisions={"education": "no Phd", "startup": "start up"})))
print("Expected Utility for not doing a Phd + no startup: {}".format(
    ve.expected_utility(decisions={"education": "no Phd", "startup": "no start up"})))
print "Get optimal decision for education using max_sum Algorithm: {}".format(ve.max_sum("education"))
print "Get optimal decision for startup using max_sum Algorithm: {}".format(ve.max_sum("startup"))

print ""
print "Fever Example 4.3.5 from Bayesian Artificial Intelligence - Korb Nicholson"
print ""

d0 = d0_net()

flu_0 = DiscreteNode("flu_0", values=["True", "False"])
fever_0 = DiscreteNode("fever_0", values=["True", "False"])
therm_0 = DiscreteNode("therm_0", values=["True", "False"])
take_aspirin_0 = DecisionNode("take_aspirin_0", decisions=["Yes", "No"])
fever_later_0 = DiscreteNode("fever_later_0", values=["True", "False"])
reaction_0 = DiscreteNode("reaction_0", values=["Yes", "No"])
utility_0 = UtilityNode("utility_0")

d0.add_nodes([flu_0, fever_0, therm_0, take_aspirin_0, fever_later_0, reaction_0, utility_0])

d0.add_edge(flu_0, fever_0)
d0.add_edge(fever_0, therm_0)
d0.add_edge(fever_0, fever_later_0)
d0.add_edge(fever_later_0, utility_0)
d0.add_edge(take_aspirin_0, reaction_0)
d0.add_edge(take_aspirin_0, fever_later_0)
d0.add_edge(reaction_0, utility_0)

flu_0.set_probability("True", 0.05)
flu_0.set_probability("False", 0.95)

fever_0.set_probability("True", 0.95, parentValues={"flu_0": "True"})
fever_0.set_probability("True", 0.02, parentValues={"flu_0": "False"})
fever_0.set_probability("False", 0.05, parentValues={"flu_0": "True"})
fever_0.set_probability("False", 0.98, parentValues={"flu_0": "False"})

therm_0.set_probability("True", 0.90, parentValues={"fever_0": "True"})
therm_0.set_probability("True", 0.05, parentValues={"fever_0": "False"})
therm_0.set_probability("False", 0.10, parentValues={"fever_0": "True"})
therm_0.set_probability("False", 0.95, parentValues={"fever_0": "False"})

fever_later_0.set_probability("True", 0.05, parentValues={"fever_0": "True", "take_aspirin_0": "Yes"})
fever_later_0.set_probability("True", 0.90, parentValues={"fever_0": "True", "take_aspirin_0": "No"})
fever_later_0.set_probability("True", 0.01, parentValues={"fever_0": "False", "take_aspirin_0": "Yes"})
fever_later_0.set_probability("True", 0.02, parentValues={"fever_0": "False", "take_aspirin_0": "No"})
fever_later_0.set_probability("False", 0.95, parentValues={"fever_0": "True", "take_aspirin_0": "Yes"})
fever_later_0.set_probability("False", 0.10, parentValues={"fever_0": "True", "take_aspirin_0": "No"})
fever_later_0.set_probability("False", 0.99, parentValues={"fever_0": "False", "take_aspirin_0": "Yes"})
fever_later_0.set_probability("False", 0.98, parentValues={"fever_0": "False", "take_aspirin_0": "No"})

reaction_0.set_probability("Yes", 0.05, parentValues={"take_aspirin_0": "Yes"})
reaction_0.set_probability("Yes", 0.00, parentValues={"take_aspirin_0": "No"})
reaction_0.set_probability("No", 0.95, parentValues={"take_aspirin_0": "Yes"})
reaction_0.set_probability("No", 1.0, parentValues={"take_aspirin_0": "No"})

utility_0.set_utility(-50, {"fever_later_0": "True", "reaction_0": "Yes"})
utility_0.set_utility(-10, {"fever_later_0": "True", "reaction_0": "No"})
utility_0.set_utility(-30, {"fever_later_0": "False", "reaction_0": "Yes"})
utility_0.set_utility(50, {"fever_later_0": "False", "reaction_0": "No"})

d0.set_partial_ordering([["flu_0", "fever_0", "therm_0"], "take_aspirin_0", ["fever_later_0", "reaction_0"]])

ve = VariableElimination(d0)

print("Expected Utility for Taking Aspirin: {}".format(ve.expected_utility(decisions={"take_aspirin_0": "Yes"})))
print("Expected Utility for not Taking Aspirin: {}".format(ve.expected_utility(decisions={"take_aspirin_0": "No"})))
print "Get optimal decision using classic algorithm: ", ve.get_optimal_decisions(["take_aspirin_0"])
print "Get optimal decision for education using max_sum Algorithm: {}".format(ve.max_sum("take_aspirin_0"))

print ""
print "L2Tor example"
print ""

d0 = d0_net()

skill_0 = DiscreteNode("skill_0", values=[0, 1, 2, 3, 4, 5])
action_0 = DecisionNode("action_0", decisions=["1", "2", "3", "4"])
observation_0 = DiscreteNode("observation_0", values=["+O", "-O"])

d0.add_nodes([skill_0, action_0, observation_0])
d0.add_edge(skill_0, observation_0)
d0.add_edge(skill_0, action_0)
d0.add_edge(action_0, observation_0)

d0.set_partial_ordering(["skill_0", "action_0", "observation_0"])

skill_0.set_cpd([1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6])

action_0.set_cpd([[0.40, 0.30, 0.20, 0.15, 0.10, 0.05],
                  [0.30, 0.35, 0.40, 0.25, 0.25, 0.25],
                  [0.25, 0.25, 0.25, 0.40, 0.35, 0.30],
                  [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]])

action_0.set_action_norm({'1': {'loc': .15, 'scale': .05},
                          '2': {'loc': .40, 'scale': .05},
                          '3': {'loc': .625, 'scale': .05},
                          '4': {'loc': .85, 'scale': .05}})

observation_0.set_cpd([[[0.50, 0.3, 0.25, 0.15],
                        [0.55, 0.40, 0.30, 0.20],
                        [0.65, 0.55, 0.40, 0.30],
                        [0.75, 0.65, 0.50, 0.40],
                        [0.85, 0.75, 0.60, 0.50],
                        [0.95, 0.85, 0.70, 0.60]],
                       [[0.50, 0.67, 0.75, 0.85],
                        [0.45, 0.60, 0.70, 0.80],
                        [0.35, 0.45, 0.60, 0.70],
                        [0.25, 0.35, 0.50, 0.60],
                        [0.15, 0.25, 0.40, 0.50],
                        [0.05, 0.15, 0.30, 0.40]]])

two_tdn = Two_TDN()

skill_t = DiscreteNode("skill_t", values=[0, 1, 2, 3, 4, 5])

action_t = DecisionNode("action_t", decisions=["1", "2", "3", "4"])
observation_t = DiscreteNode("observation_t", values=["+O", "-O"])

skill_t_plus_one = DiscreteNode("skill_t_plus_one", values=[0, 1, 2, 3, 4, 5])

action_t_plus_one = DecisionNode("action_t_plus_one", decisions=["1", "2", "3", "4"])
observation_t_plus_one = DiscreteNode("observation_t_plus_one", values=["+O", "-O"])

transition_dict = {'+O': {"1": {0: {0: 0.60, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.25, 1: 0.70, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                2: {0: 0.15, 1: 0.20, 2: 0.80, 3: 0.00, 4: 0.00, 5: 0.00},
                                3: {0: 0.00, 1: 0.10, 2: 0.15, 3: 0.85, 4: 0.00, 5: 0.00},
                                4: {0: 0.00, 1: 0.00, 2: 0.05, 3: 0.10, 4: 0.90, 5: 0.00},
                                5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.05, 4: 0.10, 5: 1.00}
                                },
                          "2": {0: {0: 0.55, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.35, 1: 0.60, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                2: {0: 0.10, 1: 0.30, 2: 0.70, 3: 0.00, 4: 0.00, 5: 0.00},
                                3: {0: 0.00, 1: 0.10, 2: 0.20, 3: 0.70, 4: 0.00, 5: 0.00},
                                4: {0: 0.00, 1: 0.00, 2: 0.10, 3: 0.25, 4: 0.75, 5: 0.00},
                                5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.05, 4: 0.25, 5: 1.00}
                                },
                          "3": {0: {0: 0.40, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.40, 1: 0.40, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                2: {0: 0.20, 1: 0.40, 2: 0.40, 3: 0.00, 4: 0.00, 5: 0.00},
                                3: {0: 0.00, 1: 0.20, 2: 0.40, 3: 0.50, 4: 0.00, 5: 0.00},
                                4: {0: 0.00, 1: 0.00, 2: 0.20, 3: 0.40, 4: 0.60, 5: 0.00},
                                5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.10, 4: 0.40, 5: 1.00}
                                },
                          "4": {0: {0: 0.30, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.45, 1: 0.30, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
                                2: {0: 0.20, 1: 0.45, 2: 0.30, 3: 0.00, 4: 0.00, 5: 0.00},
                                3: {0: 0.05, 1: 0.20, 2: 0.45, 3: 0.30, 4: 0.00, 5: 0.00},
                                4: {0: 0.00, 1: 0.05, 2: 0.20, 3: 0.45, 4: 0.40, 5: 0.00},
                                5: {0: 0.00, 1: 0.00, 2: 0.05, 3: 0.25, 4: 0.60, 5: 1.00}
                                }
                          },
                   '-O': {"1": {0: {0: 1.00, 1: 0.10, 2: 0.05, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.00, 1: 0.90, 2: 0.10, 3: 0.15, 4: 0.00, 5: 0.00},
                                2: {0: 0.00, 1: 0.00, 2: 0.85, 3: 0.25, 4: 0.20, 5: 0.00},
                                3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.60, 4: 0.30, 5: 0.25},
                                4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.50, 5: 0.35},
                                5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.40}
                                },
                          "2": {0: {0: 1.00, 1: 0.25, 2: 0.05, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.00, 1: 0.75, 2: 0.25, 3: 0.10, 4: 0.00, 5: 0.00},
                                2: {0: 0.00, 1: 0.00, 2: 0.70, 3: 0.25, 4: 0.10, 5: 0.00},
                                3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.65, 4: 0.30, 5: 0.10},
                                4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.60, 5: 0.35},
                                5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.55}
                                },
                          "3": {0: {0: 1.00, 1: 0.20, 2: 0.20, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.00, 1: 0.80, 2: 0.40, 3: 0.20, 4: 0.00, 5: 0.00},
                                2: {0: 0.00, 1: 0.00, 2: 0.40, 3: 0.40, 4: 0.20, 5: 0.00},
                                3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.40, 4: 0.40, 5: 0.20},
                                4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.40, 5: 0.40},
                                5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.40}
                                },
                          "4": {0: {0: 1.00, 1: 0.40, 2: 0.15, 3: 0.00, 4: 0.00, 5: 0.00},
                                1: {0: 0.00, 1: 0.60, 2: 0.35, 3: 0.15, 4: 0.00, 5: 0.00},
                                2: {0: 0.00, 1: 0.00, 2: 0.50, 3: 0.35, 4: 0.15, 5: 0.00},
                                3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.50, 4: 0.35, 5: 0.15},
                                4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.50, 5: 0.35},
                                5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.50}
                                }
                          }
                   }

two_tdn.add_nodes([skill_t, action_t, observation_t, skill_t_plus_one, action_t_plus_one, observation_t_plus_one])

two_tdn.add_intra_edge(skill_t, action_t)
two_tdn.add_intra_edge(skill_t, observation_t)
two_tdn.add_intra_edge(action_t, observation_t)

two_tdn.add_inter_edge(skill_t, skill_t_plus_one)
two_tdn.add_inter_edge(action_t, skill_t_plus_one)
two_tdn.add_inter_edge(observation_t, skill_t_plus_one)


# This is just a helper function to translate the given dictionary to an array (for this specific example.
# Should not be applied to different examples.

def convert_to_array(transition_):
    if isinstance(transition_, dict):
        cpd = []

        def get_table(transition, answer, action):
            return pd.DataFrame(transition[answer][action])

        for i in transition_.keys():
            temp = []
            for j in transition_[i]:
                temp.append(get_table(transition_, i, j).as_matrix())

            cpd.append(temp)

        # return np.transpose(np.array(cpd).T, (0, 1, 3, 2))
        return np.array(cpd).T


print(convert_to_array(transition_dict).shape)
two_tdn.add_transition(skill_t_plus_one, convert_to_array(transition_dict))

DDN = DynamicDecisionNetwork(d0, two_tdn)
new_net = DDN.unroll(2)

for i in new_net.get_all_nodes():
    print "Current One:"
    print i.name
    print(i.cpd)

# ve = VariableElimination(new_net)
# print("Optimal decision: ", ve.get_optimal_decisions(["action_0"]))
# print("Get optimal decision for action_0 using max_sum Algorithm: {}".format(ve.max_sum("action_0")))

# mu = [.15, .40, .625, .85]
# variance = [.05, .05, .05, .05]
# sigma = [sqrt(i) for i in variance]
# x = [np.linspace(mu[i] - 3 * sigma[i], mu[i] + 3 * sigma[i], 100) for i, _ in enumerate(mu)]
# for i, v in enumerate(x):
#     plt.plot(v, mlab.normpdf(v, mu[i], sigma[i]))
# plt.show()
