#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:59:32 2017

@author: jpoeppel
"""

from primo2.networks import DecisionNetwork, DynamicDecisionNetwork, d0_net, Two_TDN
from primo2.nodes import DiscreteNode, DecisionNode, UtilityNode

from primo2.inference.decision import VariableElimination

"""
PHD example 7.3 from Bayesian Reasoning and Machine Learning - Barber
"""
# d0 = d0_net()
#
# education_0 = DecisionNode("education_0", decisions=["do Phd", "no Phd"]) #E
#
# income_0 = DiscreteNode("income_0", values=["low", "average", "high"]) #I
# nobel_0 = DiscreteNode("nobel_0", values=["prize", "no prize"]) #P
#
# costs_0 = UtilityNode("costs_0") #UC
# gains_0 = UtilityNode("gains_0") #UB
#
# d0.add_nodes([education_0,income_0,nobel_0,costs_0,gains_0])
#
# d0.set_zero_timeslice([income_0,nobel_0])
#
# #Add edges. Edges can either be acutal dependencies or information links.
# #The type is figured out by the nodes themsevles
# d0.add_edge(education_0, costs_0)
# d0.add_edge(education_0, nobel_0)
# d0.add_edge(education_0, income_0)
#
# d0.add_edge(nobel_0, income_0)
# d0.add_edge(income_0, gains_0)
#
# #Define CPTs: (Needs to be done AFTER the structure is defined as that)
# #determines the table structure for the different nodes
#
# income_0.set_probability("low", 0.1, parentValues={"education_0":"do Phd", "nobel_0":"no prize"})
# income_0.set_probability("low", 0.2, parentValues={"education_0":"no Phd", "nobel_0":"no prize"})
# income_0.set_probability("low", 0.01, parentValues={"education_0":"do Phd", "nobel_0":"prize"})
# income_0.set_probability("low", 0.01, parentValues={"education_0":"no Phd", "nobel_0":"prize"})
#
# income_0.set_probability("average", 0.5, parentValues={"education_0":"do Phd", "nobel_0":"no prize"})
# income_0.set_probability("average", 0.6, parentValues={"education_0":"no Phd", "nobel_0":"no prize"})
# income_0.set_probability("average", 0.04, parentValues={"education_0":"do Phd", "nobel_0":"prize"})
# income_0.set_probability("average", 0.04, parentValues={"education_0":"no Phd", "nobel_0":"prize"})
#
# income_0.set_probability("high", 0.4, parentValues={"education_0":"do Phd", "nobel_0":"no prize"})
# income_0.set_probability("high", 0.2, parentValues={"education_0":"no Phd", "nobel_0":"no prize"})
# income_0.set_probability("high", 0.95, parentValues={"education_0":"do Phd", "nobel_0":"prize"})
# income_0.set_probability("high", 0.95, parentValues={"education_0":"no Phd", "nobel_0":"prize"})
#
#
# nobel_0.set_probability("prize", 0.0000001, parentValues={"education_0":"no Phd"})
# nobel_0.set_probability("prize", 0.001, parentValues={"education_0":"do Phd"})
#
# nobel_0.set_probability("no prize", 0.9999999, parentValues={"education_0":"no Phd"})
# nobel_0.set_probability("no prize", 0.999, parentValues={"education_0":"do Phd"})
#
#
# #Define utilities
#
# costs_0.set_utility(-50000, parentValues={"education_0":"do Phd"})
# costs_0.set_utility(0, parentValues={"education_0":"no Phd"})
#
# gains_0.set_utility(100000, parentValues={"income_0":"low"})
# gains_0.set_utility(200000, parentValues={"income_0":"average"})
# gains_0.set_utility(500000, parentValues={"income_0":"high"})
#
# d0.set_partial_ordering(["education_0", ["income_0", "nobel_0"]])
#
# # print(d0.get_all_nodes())
#
# two_tdn = Two_TDN()
#
# education_t = DecisionNode("education_t", decisions=["do Phd", "no Phd"])
#
# income_t = DiscreteNode("income_t", values=["low", "average", "high"])
# income_t_plus_1 = DiscreteNode("income_t_plus_1", values=["low", "average", "high"])
#
# nobel_t = DiscreteNode("nobel_t", values=["prize", "no prize"])
# nobel_t_plus_1 = DiscreteNode("nobel_t_plus_1", values=["prize", "no prize"])
#
# costs_t = UtilityNode("costs_t")
#
# gains_t = UtilityNode("gains_t")
#
#
#
# #Add nodes to network. They can be treated the same
# two_tdn.add_nodes([education_t, income_t, income_t_plus_1, nobel_t, nobel_t_plus_1, costs_t, gains_t])
# two_tdn.set_t_timeslice([income_t, nobel_t])
# two_tdn.set_t_plus_one_timeslice([income_t_plus_1, nobel_t_plus_1])
#
# two_tdn.add_inter_edge(income_t, income_t_plus_1)
# two_tdn.add_inter_edge(nobel_t, nobel_t_plus_1)
#
# two_tdn.add_intra_edge(education_t, costs_t)
# two_tdn.add_intra_edge(education_t, nobel_t)
# two_tdn.add_intra_edge(education_t, income_t)
#
# two_tdn.add_intra_edge(income_t, gains_t)
#
# DDN = DynamicDecisionNetwork(d0, two_tdn)
#
# new_net = DDN.unroll(2)


# print "checking parents \n"
# for i in new_net.get_all_node_names():
#     print "Parent of"
#     print i
#     for j in new_net.node_lookup[i].parents:
#         print "is:"
#         print j

# print(new_net.get_partial_ordering())
# for i in new_net.get_all_nodes():
#     print i.cpd
# education_.set_decision("do phd", 0.5, parentValues={"education":"do phd"})
# education_.set_decision("do phd", 0.5, parentValues={"education":"no phd"})
# education_.set_decision("no phd", 0.5, parentValues={"education":"do phd"})
# education_.set_decision("no phd", 0.5, parentValues={"education":"no phd"})

# income_.set_probability("low", 0.33, parentValues={"income":"low"})
# income_.set_probability("low", 0.33, parentValues={"income":"average"})
# income_.set_probability("low", 0.33, parentValues={"income":"high"})
# income_.set_probability("average", 0.33, parentValues={"income":"low"})
# income_.set_probability("average", 0.33, parentValues={"income":"average"})
# income_.set_probability("average", 0.33, parentValues={"income":"high"})
# income_.set_probability("high", 0.33, parentValues={"income":"low"})
# income_.set_probability("high", 0.33, parentValues={"income":"average"})
# income_.set_probability("high", 0.33, parentValues={"income":"high"})

# nobel_.set_probability("prize", 0.5, parentValues={"nobel":"prize"})
# nobel_.set_probability("prize", 0.5, parentValues={"nobel":"no prize"})
# nobel_.set_probability("no prize", 0.5, parentValues={"nobel":"prize"})
# nobel_.set_probability("no prize", 0.5, parentValues={"nobel":"no prize"})

# costs_.set_utility(0, parentValues={"costs":"do Phd"})
# costs_.set_utility(0, parentValues={"education":"no Phd"})


# ve = VariableElimination(new_net)

# print("Expected Utility for doing a Phd: {}".format(ve.expected_utility(decisions={"education_0":"do Phd"})))
# print("Expected Utility for not doing a Phd: {}".format(ve.expected_utility(decisions={"education_0":"no Phd"})))
# print("Optimal decision: ", ve.get_optimal_decisions(["education_0"]))
# print("Get optimal decision for education using max_sum Algorithm: {}".format(ve.max_sum("education_0")))


"""
PHD + Startup example 7.4 from Bayesian Reasoning and Machine Learning - Barber
"""

# net = DecisionNetwork()


# education = DecisionNode("education", decisions=["do Phd", "no Phd"]) #E
# startup = DecisionNode("startup", decisions=["start up", "no start up"]) # S

# income = DiscreteNode("income", values=["low", "average", "high"]) #I
# nobel = DiscreteNode("nobel", values=["prize", "no prize"]) #P

# costsEducation = UtilityNode("costsE") #UC
# costsStartUp = UtilityNode("costsS") #US
# gains = UtilityNode("gains") #UB


# #Add nodes to network. They can be treated the same
# net.add_node(education)
# net.add_node(startup)
# net.add_node(income)
# net.add_node(nobel)

# net.add_node(costsEducation)
# net.add_node(costsStartUp)
# net.add_node(gains)


# #Add edges. Edges can either be acutal dependencies or information links.
# #The type is figured out by the nodes themsevles
# net.add_edge(education, costsEducation)
# net.add_edge(education, nobel)

# net.add_edge(startup, income)
# net.add_edge(startup, costsStartUp)

# net.add_edge(nobel, income)
# net.add_edge(income, gains)

# #Define CPTs: (Needs to be done AFTER the structure is defined as that)
# #determines the table structure for the different nodes

# income.set_probability("low", 0.1, parentValues={"startup":"start up", "nobel":"no prize"})
# income.set_probability("low", 0.2, parentValues={"startup":"no start up", "nobel":"no prize"})
# income.set_probability("low", 0.005, parentValues={"startup":"start up", "nobel":"prize"})
# income.set_probability("low", 0.05, parentValues={"startup":"no start up", "nobel":"prize"})

# income.set_probability("average", 0.5, parentValues={"startup":"start up", "nobel":"no prize"})
# income.set_probability("average", 0.6, parentValues={"startup":"no start up", "nobel":"no prize"})
# income.set_probability("average", 0.005, parentValues={"startup":"start up", "nobel":"prize"})
# income.set_probability("average", 0.15, parentValues={"startup":"no start up", "nobel":"prize"})

# income.set_probability("high", 0.4, parentValues={"startup":"start up", "nobel":"no prize"})
# income.set_probability("high", 0.2, parentValues={"startup":"no start up", "nobel":"no prize"})
# income.set_probability("high", 0.99, parentValues={"startup":"start up", "nobel":"prize"})
# income.set_probability("high", 0.8, parentValues={"startup":"no start up", "nobel":"prize"})


# nobel.set_probability("prize", 0.0000001, parentValues={"education":"no Phd"})
# nobel.set_probability("prize", 0.001, parentValues={"education":"do Phd"})

# nobel.set_probability("no prize", 0.9999999, parentValues={"education":"no Phd"})
# nobel.set_probability("no prize", 0.999, parentValues={"education":"do Phd"})


# #Define utilities

# costsEducation.set_utility(-50000, parentValues={"education":"do Phd"})
# costsEducation.set_utility(0, parentValues={"education":"no Phd"})

# costsStartUp.set_utility(-200000, parentValues={"startup":"start up"})
# costsStartUp.set_utility(0, parentValues={"startup":"no start up"})

# gains.set_utility(100000, parentValues={"income":"low"})
# gains.set_utility(200000, parentValues={"income":"average"})
# gains.set_utility(500000, parentValues={"income":"high"})

# net.set_PartialOrdering([education, nobel, startup, income])
# ve = VariableElimination(net)

# print("Expected Utility for doing a Phd + startup: {}".format(ve.expected_utility(decisions={"education":"do Phd", "startup": "start up"})))
# print("Expected Utility for doing a Phd + no startup: {}".format(ve.expected_utility(decisions={"education":"do Phd", "startup": "no start up"})))
# print("Expected Utility for not doing a Phd + startup: {}".format(ve.expected_utility(decisions={"education":"no Phd", "startup": "start up"})))
# print("Expected Utility for not doing a Phd + no startup: {}".format(ve.expected_utility(decisions={"education":"no Phd", "startup": "no start up"})))
# print("Get optimal decision for education using max_sum Algorithm: {}".format(ve.max_sum("education")))
# print("Get optimal decision for startup using max_sum Algorithm: {}".format(ve.max_sum("startup")))
# print("Optimal deciosn: ", ve.get_optimal_decisions(["startup", "education"]))

# net2 = net.copy()


# Define CPTs: (Needs to be done AFTER the structure is defined as that)
# determines the table structure for the different nodes


# net.set_PartialOrdering([education, [income,nobel]])
# print(net.get_all_nodes())


# d_net = DynamicBayesianNetwork(b0=net, two_tdn=net2)
# print d_net.b0.get_all_node_names()
# print d_net.two_tdn.get_all_node_names()
# d_net.unroll(3)
# d_net = DynamicBayesianNetwork(net)

# print "What?"
# print d_net.b0.get_all_nodes()

# education_ = DecisionNode("education_", decisions=["do Phd", "no Phd"]) #E

# income_ = DiscreteNode("income_", values=["low", "average", "high"]) #I
# nobel_ = DiscreteNode("nobel_", values=["prize", "no prize"]) #P

# costs_ = UtilityNode("costs_") #UC
# gains_ = UtilityNode("gains_") #UB


# d_net.two_tbn.add_node(education_)
# d_net.two_tbn.add_node(income_)
# d_net.two_tbn.add_node(nobel_)

# d_net.two_tbn.add_node(costs_)
# d_net.two_tbn.add_node(gains_)

# income_.set_probability("low", 0.11, parentValues={"income":"low"})
# income_.set_probability("low", 0.21, parentValues={"income":"low"})
# income_.set_probability("low", 0.011, parentValues={"income":"low"})
# income_.set_probability("low", 0.011, parentValues={"income":"low"})

# income_.set_probability("average", 0.51, parentValues={"income":"average"})
# income_.set_probability("average", 0.61, parentValues={"income":"average"})
# income_.set_probability("average", 0.041, parentValues={"income":"average"})
# income_.set_probability("average", 0.041, parentValues={"income":"average"})

# income_.set_probability("high", 0.41, parentValues={"income":"high"})
# income_.set_probability("high", 0.21, parentValues={"income":"high"})
# income_.set_probability("high", 0.951, parentValues={"income":"high"})
# income_.set_probability("high", 0.951, parentValues={"income":"high"})


# nobel_.set_probability("prize", 0.00000011, parentValues={"nobel":"prize"})
# nobel_.set_probability("prize", 0.0011, parentValues={"nobel":"prize"})

# nobel_.set_probability("no prize", 0.99999991, parentValues={"nobel":"no prize"})
# nobel_.set_probability("no prize", 0.9991, parentValues={"nobel":"no prize"})


# #Define utilities

# costs_.set_utility(-50001, parentValues={"costs"})
# costs_.set_utility(1, parentValues={"costs"})

# gains_.set_utility(100001, parentValues={"gains"})
# gains_.set_utility(200001, parentValues={"gains"})
# gains_.set_utility(500001, parentValues={"gains"})


# ve = VariableElimination(net)

# print("Expected Utility for doing a Phd: {}".format(ve.expected_utility(decisions={"education":"do Phd"})))
# print("Expected Utility for not doing a Phd: {}".format(ve.expected_utility(decisions={"education":"no Phd"})))
# print("Optimal deciosn: ", ve.get_optimal_decisions(["education"]))
# ccprint("Get optimal decision for education using max_sum Algorithm: {}".format(ve.max_sum("education")))

# ve = VariableElimination(d_net._b0)
# print("Get optimal decision for startup using max_sum Algorithm: {}".format(ve.max_sum("startup")))


# """
# Fever Example 4.3.5 from Bayesian Artificial Intelligence - Korb Nicholson
# """
#
# d0 = d0_net()
#
# flu_0 = DiscreteNode("flu_0", values=["True", "False"])
# fever_0 = DiscreteNode("fever_0", values=["True", "False"])
# therm_0 = DiscreteNode("therm_0", values=["True", "False"])
# take_aspirin_0 = DecisionNode("take_aspirin_0", decisions=["Yes", "No"])
# fever_later_0 = DiscreteNode("fever_later_0", values=["True", "False"])
# reaction_0 = DiscreteNode("reaction_0", values=["Yes", "No"])
# utility_0 = UtilityNode("utility_0")
#
# d0.add_nodes([flu_0, fever_0, therm_0, take_aspirin_0, fever_later_0, reaction_0, utility_0])
#
# d0.add_edge(flu_0, fever_0)
# d0.add_edge(fever_0, therm_0)
# d0.add_edge(fever_0, fever_later_0)
# d0.add_edge(fever_later_0, utility_0)
# d0.add_edge(take_aspirin_0, reaction_0)
# d0.add_edge(take_aspirin_0, fever_later_0)
# d0.add_edge(reaction_0, utility_0)
#
# flu_0.set_probability("True", 0.05)
# flu_0.set_probability("False", 0.95)
#
# fever_0.set_probability("True", 0.95, parentValues={"flu_0": "True"})
# fever_0.set_probability("True", 0.02, parentValues={"flu_0": "False"})
# fever_0.set_probability("False", 0.05, parentValues={"flu_0": "True"})
# fever_0.set_probability("False", 0.98, parentValues={"flu_0": "False"})
#
# therm_0.set_probability("True", 0.90, parentValues={"fever_0": "True"})
# therm_0.set_probability("True", 0.05, parentValues={"fever_0": "False"})
# therm_0.set_probability("False", 0.10, parentValues={"fever_0": "True"})
# therm_0.set_probability("False", 0.95, parentValues={"fever_0": "False"})
#
# fever_later_0.set_probability("True", 0.05, parentValues={"fever_0": "True", "take_aspirin_0": "Yes"})
# fever_later_0.set_probability("True", 0.90, parentValues={"fever_0": "True", "take_aspirin_0": "No"})
# fever_later_0.set_probability("True", 0.01, parentValues={"fever_0": "False", "take_aspirin_0": "Yes"})
# fever_later_0.set_probability("True", 0.02, parentValues={"fever_0": "False", "take_aspirin_0": "No"})
# fever_later_0.set_probability("False", 0.95, parentValues={"fever_0": "True", "take_aspirin_0": "Yes"})
# fever_later_0.set_probability("False", 0.10, parentValues={"fever_0": "True", "take_aspirin_0": "No"})
# fever_later_0.set_probability("False", 0.99, parentValues={"fever_0": "False", "take_aspirin_0": "Yes"})
# fever_later_0.set_probability("False", 0.98, parentValues={"fever_0": "False", "take_aspirin_0": "No"})
#
# reaction_0.set_probability("Yes", 0.05, parentValues={"take_aspirin_0": "Yes"})
# reaction_0.set_probability("Yes", 0.00, parentValues={"take_aspirin_0": "No"})
# reaction_0.set_probability("No", 0.95, parentValues={"take_aspirin_0": "Yes"})
# reaction_0.set_probability("No", 1.0, parentValues={"take_aspirin_0": "No"})
#
# utility_0.set_utility(-50, {"fever_later_0": "True", "reaction_0": "Yes"})
# utility_0.set_utility(-10, {"fever_later_0": "True", "reaction_0": "No"})
# utility_0.set_utility(-30, {"fever_later_0": "False", "reaction_0": "Yes"})
# utility_0.set_utility(50,  {"fever_later_0": "False", "reaction_0": "No"})
#
# d0.set_partial_ordering([["flu_0", "fever_0", "therm_0"], "take_aspirin_0", ["fever_later_0", "reaction_0"]])
#
# ve = VariableElimination(d0)
#
# print("Expected Utility for Taking Aspirin: {}".format(ve.expected_utility(decisions={"take_aspirin_0": "Yes"})))
# print("Expected Utility for not Taking Aspirin: {}".format(ve.expected_utility(decisions={"take_aspirin_0": "No"})))
# print("Optimal decision: ", ve.get_optimal_decisions(["take_aspirin_0"]))
# print("Get optimal decision for education using max_sum Algorithm: {}".format(ve.max_sum("take_aspirin_0")))


# L2Tor example

d0 = d0_net()

skill_0 = DiscreteNode("skill_0", values=[0, 1, 2, 3, 4, 5])
action_0 = DecisionNode("action_0", decisions=["1", "2", "3", "4"])
observation_0 = DiscreteNode("observation_0", values=["+O", "-O"])

d0.add_nodes([skill_0, action_0, observation_0])
d0.add_edge(skill_0, observation_0)
d0.add_edge(skill_0, action_0)
d0.add_edge(action_0, observation_0)

d0.set_zero_timeslice([skill_0, observation_0, action_0])

d0.set_partial_ordering(["skill_0", "action_0", "observation_0"])

skill_0.set_cpd([1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6])

action_0.set_cpd([[0.40, 0.30, 0.20, 0.15, 0.10, 0.05],
                  [0.30, 0.35, 0.40, 0.25, 0.25, 0.25],
                  [0.25, 0.25, 0.25, 0.40, 0.35, 0.30],
                  [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]])

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

cpd = [[[[0.60, 1.00], [0.55, 1.00], [0.40, 1.00], [0.30, 1.00]],
        [[0.25, 0], [0.35, 0], [0.40, 0], [0.45, 0]],
        [[0.15, 0], [0.10, 0], [0.20, 0], [0.20, 0]],
        [[0, 0], [0, 0], [0, 0], [0.05, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]]],
       [[[0, 0.10], [0, 0.25], [0, 0.20], [0, 0.40]],
        [[0.70, 0.90], [0.60, 0.75], [0.40, 0.80], [0.30, 0.60]],
        [[0.20, 0], [0.30, 0], [0.40, 0], [0.45, 0]],
        [[0.10, 0], [0.10, 0], [0.20, 0], [0.20, 0]],
        [[0, 0], [0, 0], [0, 0], [0.05, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]]],
       [[[0, 0.05], [0, 0.05], [0, 0.20], [0, 0.15]],
        [[0, 0.10], [0, 0.25], [0, 0.40], [0, 0.35]],
        [[0, 0.25], [0, 0.25], [0, 0.40], [0, 0.35]],
        [[0.85, 0.60], [0.70, 0.65], [0.50, 0.40], [0.30, 0.50]],
        [[0.10, 0], [0.25, 0], [0.40, 0], [0.45, 0]],
        [[0.05, 0], [0.05, 0], [0.10, 0], [0.25, 0]]],
       [[[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0.15], [0, 0.10], [0, 0.20], [0, 0.15]],
        [[0, 0.25], [0, 0.25], [0, 0.40], [0, 0.35]],
        [[0.85, 0.60], [0.70, 0.65], [0.50, 0.40], [0.30, 0.35]],
        [[0.10, 0], [0.25, 0], [0.40, 0], [0.45, 0.]],
        [[0.05, 0], [0.05, 0], [0.10, 0], [0.25, 0]]],
       [[[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0.25], [0, 0.10], [0, 0.20], [0, 0.15]],
        [[0, 0.35], [0, 0.35], [0, 0.40], [0, 0.35]],
        [[1.0, 0.40], [1.0, 0.55], [1.0, 0.40], [1.0, 0.50]]],
       [[[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0.15], [0, 0.10], [0, 0.20], [0, 0.15]],
        [[0, 0.25], [0, 0.25], [0, 0.40], [0, 0.35]],
        [[0.85, 0.60], [0.70, 0.65], [0.50, 0.40], [0.30, 0.35]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]]]]

two_tdn.add_nodes([skill_t, action_t, observation_t, skill_t_plus_one, action_t_plus_one, observation_t_plus_one])
two_tdn.set_t_timeslice([skill_t, action_t, observation_t])
two_tdn.set_t_plus_one_timeslice([skill_t_plus_one, action_t_plus_one, observation_t_plus_one])

two_tdn.add_intra_edge(skill_t, action_t)
two_tdn.add_intra_edge(skill_t, observation_t)
two_tdn.add_intra_edge(action_t, observation_t)

two_tdn.add_inter_edge(skill_t, skill_t_plus_one)
two_tdn.add_inter_edge(action_t, skill_t_plus_one)
two_tdn.add_inter_edge(observation_t, skill_t_plus_one)

two_tdn.add_transition(skill_t_plus_one, cpd)


# two_tdn.set_transition_probability(skill_t_plus_one, )

DDN = DynamicDecisionNetwork(d0, two_tdn)

new_net = DDN.unroll(2)

print new_net.node_lookup["observation_1"].cpd
print new_net.get_partial_ordering()

ve = VariableElimination(new_net)
# print("Optimal decision: ", ve.get_optimal_decisions(["action_0"]))
print("Get optimal decision for education using max_sum Algorithm: {}".format(ve.max_sum("action_1")))

# new = {'+O': {"1": {0: {0: 0.60, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.25, 1: 0.70, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     2: {0: 0.15, 1: 0.20, 2: 0.80, 3: 0.00, 4: 0.00, 5: 0.00},
#                     3: {0: 0.00, 1: 0.10, 2: 0.15, 3: 0.85, 4: 0.00, 5: 0.00},
#                     4: {0: 0.00, 1: 0.00, 2: 0.05, 3: 0.10, 4: 0.90, 5: 0.00},
#                     5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.05, 4: 0.10, 5: 1.00}
#                     },
#               "2": {0: {0: 0.55, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.35, 1: 0.60, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     2: {0: 0.10, 1: 0.30, 2: 0.70, 3: 0.00, 4: 0.00, 5: 0.00},
#                     3: {0: 0.00, 1: 0.10, 2: 0.20, 3: 0.70, 4: 0.00, 5: 0.00},
#                     4: {0: 0.00, 1: 0.00, 2: 0.10, 3: 0.25, 4: 0.75, 5: 0.00},
#                     5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.05, 4: 0.25, 5: 1.00}
#                     },
#               "3": {0: {0: 0.40, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.40, 1: 0.40, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     2: {0: 0.20, 1: 0.40, 2: 0.40, 3: 0.00, 4: 0.00, 5: 0.00},
#                     3: {0: 0.00, 1: 0.20, 2: 0.40, 3: 0.50, 4: 0.00, 5: 0.00},
#                     4: {0: 0.00, 1: 0.00, 2: 0.20, 3: 0.40, 4: 0.60, 5: 0.00},
#                     5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.10, 4: 0.40, 5: 1.00}
#                     },
#               "4": {0: {0: 0.30, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.45, 1: 0.30, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00},
#                     2: {0: 0.20, 1: 0.45, 2: 0.30, 3: 0.00, 4: 0.00, 5: 0.00},
#                     3: {0: 0.05, 1: 0.20, 2: 0.45, 3: 0.30, 4: 0.00, 5: 0.00},
#                     4: {0: 0.00, 1: 0.05, 2: 0.20, 3: 0.45, 4: 0.40, 5: 0.00},
#                     5: {0: 0.00, 1: 0.00, 2: 0.05, 3: 0.25, 4: 0.60, 5: 1.00}
#                     }
#               },
#        '-O': {"1": {0: {0: 1.00, 1: 0.10, 2: 0.05, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.00, 1: 0.90, 2: 0.10, 3: 0.15, 4: 0.00, 5: 0.00},
#                     2: {0: 0.00, 1: 0.00, 2: 0.85, 3: 0.25, 4: 0.20, 5: 0.00},
#                     3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.60, 4: 0.30, 5: 0.25},
#                     4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.50, 5: 0.35},
#                     5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.40}
#                     },
#               "2": {0: {0: 1.00, 1: 0.25, 2: 0.05, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.00, 1: 0.75, 2: 0.25, 3: 0.10, 4: 0.00, 5: 0.00},
#                     2: {0: 0.00, 1: 0.00, 2: 0.70, 3: 0.25, 4: 0.10, 5: 0.00},
#                     3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.65, 4: 0.30, 5: 0.10},
#                     4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.60, 5: 0.35},
#                     5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.55}
#                     },
#               "3": {0: {0: 1.00, 1: 0.20, 2: 0.20, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.00, 1: 0.80, 2: 0.40, 3: 0.20, 4: 0.00, 5: 0.00},
#                     2: {0: 0.00, 1: 0.00, 2: 0.40, 3: 0.40, 4: 0.20, 5: 0.00},
#                     3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.40, 4: 0.40, 5: 0.20},
#                     4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.40, 5: 0.40},
#                     5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.40}
#                     },
#               "4": {0: {0: 1.00, 1: 0.40, 2: 0.15, 3: 0.00, 4: 0.00, 5: 0.00},
#                     1: {0: 0.00, 1: 0.60, 2: 0.35, 3: 0.15, 4: 0.00, 5: 0.00},
#                     2: {0: 0.00, 1: 0.00, 2: 0.50, 3: 0.35, 4: 0.15, 5: 0.00},
#                     3: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.50, 4: 0.35, 5: 0.15},
#                     4: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.50, 5: 0.35},
#                     5: {0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.50}
#                     }
#               }
#        }
