#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:45:51 2017

@author: jpoeppel
"""

from __future__ import division

import functools

from ..nodes import UtilityNode
from .factor import Factor
from primo2.nodes import DiscreteNode, DecisionNode, UtilityNode
import numpy as np


class VariableElimination(object):

    def __init__(self, decisionNetwork):
        self.net = decisionNetwork

    def get_decision(self, decisionNode, otherDecisions=None):
        """
            
        """
        pass

    def _combine_factors(self, factors):
        """
            Helper function to combine multiple joint factors.
            
            Parameters
            ----------
            factors: iterable(tuple)
                An iterable of joint factors
                
            Returns
            -------
                tuple
                A joint factor representing the combination of all the given
                factors.
        """

        def _combine_two_factors(f1, f2):
            return (f1[0] * f2[0], f1[1] + f2[1])

        return functools.reduce(_combine_two_factors, factors)

    def _marginalize_joint_factor(self, factor, variable):
        """
            Helper function to marginalize a variable from a joint factor, 
            following eq 23.6 in Koller, Friedman: Probabilistic Graphical Models.
            
            Parameters
            ---------
            factor: tuple
                The joint factor to be reduced
                
            variable: string
                The name of the variable to be marginalized out.
                
            Returns
            -------
                tuple
                A joint factor with the given variable marginalized out.
        """
        tmp = factor[0].marginalize(variable)
        return (tmp, (factor[0] * factor[1]).marginalize(variable) / tmp)

    def generalized_VE(self, joint_factors, elimination_variables):
        """
            Generalized variable elimination for joint factors in influence
            diagrams (cf. Koller, Friedman: Probabilistic Graphical Models,
            Algorithm 23.2)
            
            Parameters
            ---------
            joint_factors: set(tuple)
                A set of tuple containing probabilistic factors and utility factors
                
            elimination_variables: list
                A list of variables that should be marginalized out
                
            Returns
            -------
                tuple
                A joint factor containing a probabilistic factor and a 
                utility factor over all the variables from the initial
                joint_factors which have not been eliminated.
        """

        working_factors = set(joint_factors)
        for var in elimination_variables:
            relevant_factors = set([f for f in working_factors if var in f[0].variableOrder])
            tmp_factor = self._combine_factors(relevant_factors)
            marginalized_factor = self._marginalize_joint_factor(tmp_factor, var)

            working_factors = (working_factors - relevant_factors)
            working_factors.add(marginalized_factor)

        res_factor = self._combine_factors(working_factors)

        return res_factor

    def expected_utility(self, decisions=None):
        """
            Computes the expected utility fo the decision network, given the 
            provided decisions. If no decisions are given, the algorithm
            assumes that the decisionNodes' state has already been set.
            
            Parameters
            ----------
            decisions: dict (optional)
                A dictionary containing the decision variable names as keys
                and their set decision as value.
                
            Returns
            -------
                float
                The expected utility for those decisions.
        """
        if decisions is None:
            decisions = {}
        # Set given decisions
        for decision in decisions:
            decisionNode = self.net.node_lookup[decision]
            #            decisionNode.clear()
            decisionNode.set_decision(decisions[decision])

        # Create joint factors
        factors = set([])
        for node in self.net.node_lookup.values():
            factors.add(Factor.joint_factor(node))

        eliminations = [node for node in self.net.node_lookup.values() if not isinstance(node, UtilityNode)]
        return self.generalized_VE(factors, eliminations)[1].potentials

    def _optimize_locally(self, decisionNodeName):
        """
            Helper function to choose the optimal decision for the given
            decisionNode. 
            
            Parameter
            ---------
            decisionNodeName: string
                The name of the decision to be optimized
                
            Returns
            -------
                string
                The optimal decision for this node
        """
        nodes = [node for node in self.net.node_lookup.keys()
                 if node != decisionNodeName and
                 not isinstance(self.net.node_lookup[node], UtilityNode) and
                 node not in self.net.node_lookup[decisionNodeName].parents]
        decisionNode = self.net.node_lookup[decisionNodeName]
        # Create joint factors
        factors = set([])
        for node in self.net.node_lookup.values():
            factors.add(Factor.joint_factor(node))

        reducedFactor = self.generalized_VE(factors, nodes)

        argmax = None
        maxVal = None
        for decision in decisionNode.values:
            eu = reducedFactor[1].get_potential({decisionNodeName: [decision]})
            if maxVal is None or eu > maxVal:
                maxVal = eu
                argmax = decision

        return argmax

    def get_optimal_decisions(self, decisionOrder, fixedDecisions=None):
        """
            Iteratively computes the optimal decisions for all given decision
            nodes according to the "Iterated optimization for influence diagrams
            with acyclic relevance graphs" 
            (cf. Koller, Friedman: Probabilistic Graphical Models, alg 23.3)
            
            Parameters
            ----------
            decisionOrder: list
                A list of decisionNode names giving the ordering according to
                the relevance graph of the network. (also known as the
                inverted partial ordering of the decisions)
                
            fixedDecisions: dict (optional)
                A dictionary which allows to optionally specify certain
                decisions which should not be optimized
                
            Returns
            -------
                dict
                A dictionary containing the optimal decision for each of the
                decisionNodes
        """
        if fixedDecisions is None:
            fixedDecisions = {}

        # Initialize random fully mixed strategy
        for decisionNode in decisionOrder:
            if not decisionNode in fixedDecisions:
                self.net.node_lookup[decisionNode].fully_mixed()
            else:
                self.net.node_lookup[decisionNode].set_decision(fixedDecisions[decisionNode])

        solution = {}
        for decisionNode in decisionOrder:
            localDecision = self._optimize_locally(decisionNode)
            solution[decisionNode] = localDecision

        return solution

    @staticmethod
    def inner_product(factors, decision_factors=None, utilities=None):
        """
            Helper Function to multiply all factors in the given list. 
            This function is guided by the Algorithm 7.3.2 at page 112 in "Bayesian Reasoning and Machine Learning"
            from David Barber.
            Here the inner_product is calculated before the summing/maxing out using factors

            
            Parameters
            ----------
            factors: list
                A list of factors to be multiplied with

            decision_factors: list
                A list of decisions to be multiplied with the factors. Not none if the decisions
                in the net have a decision rule (CPD)

            utilities: list
                A list of utilities to be multiplied with the factors


                
            Returns
            -------
                factor
                The resulting (combined) factor
        """
        prob_product = factors[0]

        for i, v in enumerate(factors):
            if i != len(factors) - 1:
                prob_product = prob_product * factors[i + 1]
        combined_factors = prob_product

        if decision_factors:
            decision_product = decision_factors[0]

            for i, v in enumerate(decision_factors):
                if i != len(decision_factors) - 1:
                    decision_product = decision_product * decision_factors[i + 1]
            combined_factors = combined_factors * decision_product

        if utilities:
            utility_factors = [Factor.from_utility_node(i) for i in utilities]
            utility_sum = utility_factors[0]

            for i, v in enumerate(utility_factors):
                if i != len(utility_factors) - 1:
                    utility_sum = utility_sum + utility_factors[i + 1]

            combined_factors = combined_factors * utility_sum

        return combined_factors

    def max_sum(self, decisionNode):
        """
            Max Sum Algorithm taken from the Algorithm 7.3.2 at page 112 in 
            "Bayesian Reasoning and Machine Learning" from David Barber.

            NEW: Now works with decision rules (That means the decisions have a CPD)

            
            Parameters
            ----------
            decisionNode: String
                The optimal decision
                
            Returns
            -------
                List
                A list containing the optimal variable and corresponding optimal utility value of the resulting max sum algorithm
        """

        partialOrder = self.net.get_partial_ordering()
        reverseOrder = partialOrder[::-1]
        randomVariables = self.net.get_random_nodes()
        utilities = self.net.get_utility_nodes()
        decisions = self.net.get_decision_nodes()

        factors = []
        for node in randomVariables:
            factors.append(Factor.from_node(node))

        decision_factors = []

        'If All decision Nodes have a decision Rule'
        if all(val.check_decision_rule() for val in decisions):
            for node in decisions:
                decision_factors.append(Factor.from_node(node))
        else:
            decision_factors = None

        current = self.inner_product(factors, decision_factors, utilities)
        for i in reverseOrder:
            if isinstance(i, list):
                if all(isinstance(self.net.node_lookup[val], DiscreteNode) for val in i) \
                        or (all(isinstance(self.net.node_lookup[val], DecisionNode) for val in i)
                            and all([val.check_decision_rule() for val in i])):
                    current = current.marginalize(i)
                else:
                    raise Exception("Marginalizing failed: Not all elements in the list are Discrete Nodes or "
                                    "Decision Nodes with a specified decision rule")
            elif isinstance(self.net.node_lookup[i], DiscreteNode):
                current = current.marginalize(i)
            elif i != decisionNode:
                current = current.maximize(i)
        return [current.values.values()[0][np.argmax(current.potentials)], max(current.potentials)]
