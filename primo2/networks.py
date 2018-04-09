#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of PRIMO2 -- Probabilistic Inference Modules.
# Copyright (C) 2013-2017 Social Cognitive Systems Group, 
#                         Faculty of Technology, Bielefeld University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public 
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

import networkx as nx

from . import exceptions
from . import nodes
from primo2.nodes import RandomNode, DiscreteNode, DecisionNode, UtilityNode


class BayesianNetwork(object):

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_lookup = {}
        self.name = ""  # Only used to be compatible with XMLBIF
        self.meta = []  # Used to be compatible with XMLBIF, stores properties 

    def add_node(self, node):
        if isinstance(node, RandomNode):
            if node.name in self.node_lookup:
                raise ValueError("The network already contains a node " \
                                 "called '{}'.".format(node.name))
            self.node_lookup[node.name] = node
            self.graph.add_node(node)
        else:
            raise TypeError("Only subclasses of RandomNode are valid nodes.")

    def remove_node(self, node):
        if node in self.graph:
            # Go over all children of this node
            for child in self.graph.succ[node]:
                child.remove_parent(self.node_lookup[node])
            self.graph.remove_node(node)
            del self.node_lookup[node]

    def remove_edge(self, from_name, to_name):
        if from_name in self.graph and to_name in self.graph:
            self.node_lookup[to_name].remove_parent(self.node_lookup[from_name])
            self.graph.remove_edge(from_name, to_name)

    def add_edge(self, from_name, to_name):
        if from_name in self.graph and to_name in self.graph:
            self.graph.add_edge(self.node_lookup[from_name], self.node_lookup[to_name])
            self.node_lookup[to_name].add_parent(self.node_lookup[from_name])
        else:
            raise Exception("Tried to add an Edge between two Nodes of " \
                            "which at least one was not contained in " \
                            "the BayesianNetwork")

    def get_node(self, node_name):
        try:
            return self.node_lookup[node_name]
        except KeyError:
            raise Exception("There is no node with name {} in the " \
                            "BayesianNetwork".format(node_name))

    def change_node_values(self, node, new_values):
        """
            Updates the values of the given node. This will invalidate the node
            and all it's children, as those nodes will expect new CPTs to match
            the new values!
            
            Parameters
            ---------
            node: String or RandomNode
                The node whose values should be changed.
            new_values: [String,]
                List of the new value names.
        """
        if node in self.node_lookup:
            self.node_lookup[node].set_values(new_values)
            for child in self.graph.succ[node]:
                child._update_dimensions()
        else:
            raise Exception("There is no node with name {} in " \
                            "the network.".format(node))

    def change_node_name(self, old_name, new_name):
        """
            Renames the given node to the new name. 
            Will have to modify all occurances of the old name.
        """
        if old_name in self.node_lookup:
            n = self.node_lookup[old_name]
            children = list(self.graph.succ[n])
            parents = list(self.graph.pred[n])
            for child in children:
                del child.parents[old_name]
                child.parents[new_name] = n
                idx = child.parentOrder.index(old_name)
                child.parentOrder[idx] = new_name

            # Fix nx graph
            self.graph.remove_node(n)

            n.name = new_name
            del self.node_lookup[old_name]
            self.node_lookup[new_name] = n

            self.graph.add_node(n)
            for child in children:
                self.graph.add_edge(n, self.node_lookup[child])
            for parent in parents:
                self.graph.add_edge(self.node_lookup[parent], n)

        else:
            raise Exception("There is no node with name {} in the " \
                            "network.".format(old_name))

    def get_all_nodes(self):
        return self.graph.nodes()

    def get_all_node_names(self):
        return self.node_lookup.keys()

    def get_nodes(self, node_names=None):
        nodes = []
        if node_names is None:
            nodes = self.graph.nodes()
        else:
            for node_name in node_names:
                nodes.append(self.get_node(node_name))
        return nodes

    def get_children(self, node_name):
        """
            Returns a list of all the children of the given node.
            
            Parameter
            --------
            node_name : String or RandomNode
                The name of the node whose children are to be returned.
                
            Returns
            -------
            [RandomNode,]
                A list containing all the nodes that have the given 
                node as parent.
        """
        return self.graph.succ[node_name]

    def get_sample(self, evidence):
        sample = {}
        for e in evidence:
            sample[e] = evidence[e]
        # Make sure evidence contains RandomNodes and not only names
        for n in nx.topological_sort(self.graph):

            if n not in evidence:
                sample[n.name] = n.sample_value(sample, self.get_children(n), forward=True)

        return sample

    def clear(self):
        """Remove all nodes and edges from the graph.
        This also removes the name, and all graph, node and edge attributes."""
        self.graph.clear()
        self.node_lookup.clear()

    def number_of_nodes(self):
        """Return the number of nodes in the graph."""
        return len(self)

    def __len__(self):
        """Return the number of nodes in the graph."""
        return len(self.graph)


# class DynamicBayesianNetwork(object):
#     """Class representing a the structure of a dynamic Bayesian network.
#
#     This temporal relationship is modelled as a 2-time-slice Bayesian
#     network (2-TBN; Koller & Friedman, 2009, §6.2.2). The Bayesian network
#     B_0 represents the initial distribution. The Bayesian network B_{->},
#     a 2-TBN, represents the process.
#     """
#
#     def __init__(self, b0=None, two_tbn=None, transitions=None):
#         """Create a dynamic Bayesian network.
#
#         Parameters
#         ----------
#         b0 : BayesianNetwork
#             The network representing the initial distribution.
#         two_tbn : BayesianNetwork
#             The two-time-slice network representing the process.
#         transitions : [(node, node_p),]
#             A list of pairs, each of which represents one transition.
#             See add_transition for more information.
#         """
#         super(DynamicBayesianNetwork, self).__init__()
#         self._b0 = BayesianNetwork() if b0 is None else b0
#         self._two_tbn = BayesianNetwork() if two_tbn is None else two_tbn
#         self._transitions = []
#         if transitions is not None:
#             self.add_transitions(transitions)
#
#     @property
#     def b0(self):
#         """Get the Bayesian network B_0.
#
#         Returns
#         -------
#         BayesianNetwork
#             The network representing the initial distribution.
#         """
#         return self._b0
#
#     @b0.setter
#     def b0(self, bn):
#         """Set the Bayesian network B_0.
#
#         Parameters
#         ----------
#         bn : BayesianNetwork
#             The network representing the initial distribution.
#         """
#         self._b0 = bn
#
#     @property
#     def two_tbn(self):
#         """Get the 2-TBN B_{->}.
#
#         Returns
#         -------
#         BayesianNetwork
#             The two-time-slice network representing the process.
#         """
#         return self._two_tbn
#
#     @two_tbn.setter
#     def two_tbn(self, bn):
#         """Set the 2-TBN B_{->}.
#
#         Parameters
#         ----------
#         bn : BayesianNetwork
#             The two-time-slice network representing the process.
#         """
#         self._two_tbn = bn
#
#     def add_transition(self, node, node_p):
#         """Add a transition connecting nodes when unrolling the network.
#
#         The transition is a directed edge from node X_i (`node`) to node X_i'
#         (`node_p`).
#
#         Parameters
#         ----------
#         node : RandomNode, String
#             The node X_i in the next time-slice.
#
#         node_p: RandomNode, String
#             The corresponding node X_i' in the current time-slice.
#
#         Raises
#         ------
#         primo2.exceptions.StructureError
#             If a node specified in the transition model cannot be found in the
#             corresponding networks.
#         """
#         if node_p not in self._b0.get_all_nodes():
#             raise exceptions.StructureError(
#                 'Node "{}" is not found in B_0 network "{}".'.format(
#                     node_p, self._b0.name))
#         if node_p not in self._two_tbn.get_all_nodes():
#             raise exceptions.StructureError(
#                 'Node "{}" is not found in B_{->} network "{}".'.format(
#                     node_p, self._two_tbn.name))
#         if node not in self._two_tbn.get_all_nodes():
#             raise exceptions.StructureError(
#                 'Node "{}" is not found in B_{{->}} network "{}".'.format(
#                     node, self._two_tbn.name))
#         self._transitions.append((node, node_p))
#
#     def add_transitions(self, transitions):
#         """Add multiple transitions connecting nodes when unrolling the network.
#
#         Parameters
#         ----------
#         transitions : [(node, node_p),]
#             A list of pairs, each of which represents one transition.
#             See add_transition for more information.
#         """
#         for transition in transitions:
#             self.add_transition(transition[0], transition[1])
#
#     @property
#     def transitions(self):
#         """Get the transition model.
#
#         Returns
#         -------
#         [(node, node_p),]
#             A list of pairs, each of which represents one transition.
#             See add_transition for more information.
#         """
#         return self._transitions
#
#
# class DecisionNetwork(object):
#
#     def __init__(self):
#         self.node_lookup = {}
#         self.partialOrdering = []
#         self.random_nodes = []
#         self.decision_nodes = []
#         self.utility_nodes = []
#
#     def add_node(self, node):
#         if isinstance(node, primo2.nodes.RandomNode):
#             if node.name in self.node_lookup.keys():
#                 raise Exception("Node name already exists in Bayesnet: " + node.name)
#             if isinstance(node, primo2.nodes.DiscreteNode):
#                 self.random_nodes.append(node)
#             elif isinstance(node, primo2.nodes.UtilityNode):
#                 self.utility_nodes.append(node)
#             elif isinstance(node, primo2.nodes.DecisionNode):
#                 self.decision_nodes.append(node)
#             else:
#                 raise Exception("Tried to add a node which the Bayesian Decision Network can not work with")
#             self.node_lookup[node.name] = node
#         else:
#             raise Exception("Can only add 'Node' and its subclasses as nodes into the BayesianNetwork")
#
#     def add_edge(self, node_from, node_to):
#
#         self.node_lookup[node_to].add_parent(self.node_lookup[node_from])
#
#     def get_PartialOrdering(self):
#         return self.partialOrdering
#
#     def set_PartialOrdering(self, partialOrder):
#         self.partialOrdering = partialOrder
#
#     def get_all_nodes(self):
#         '''Returns all RandomNodes'''
#         return self.random_nodes
#
#     def get_all_decision_nodes(self):
#         return self.decision_nodes
#
#     def get_all_utility_nodes(self):
#         return self.utility_nodes


class DynamicDecisionNetwork(object):
    """Class representing a the structure of a dynamic Decision network.
    This temporal relationship is modelled as a 2-time-slice Decision 
    network. The Decision network
    d0 represents the initial distribution. The Decision network D_{->},
    a 2-TDN, represents the process.
    """

    def __init__(self, d0=None, two_tdn=None):
        """Create a dynamic Bayesian network.
        Parameters
        ----------
        d0 : d0_net
            The network representing the initial distribution.
        two_tdn : Two_TDN
            The two-time-slice network representing the process.
        """
        super(DynamicDecisionNetwork, self).__init__()

        self._d0 = d0_net() if d0 is None else d0
        self._two_tdn = Two_TDN() if two_tdn is None else two_tdn

    def unroll(self, length):

        """Unrolling the network over specified length.
        Returns
        -------
        DecisionNetwork
            An unrolled DDN as a DecisionNetwork
        """

        unrolled_net = DecisionNetwork()

        zero_slice = self._d0.timeslice[0]
        t_plus_1_slice = self._two_tdn.timeslice['t_plus_1']
        if length > 0:
            for i in range(0, length):
                if i == 0:
                    unrolled_net.add_nodes(zero_slice + self._d0.decision_nodes + self._d0.utility_nodes)
                elif i == 1:
                    """copy nodes from the two_tdn Network into unrolled_net"""
                    unrolled_net.copy_nodes_indexed(t_plus_1_slice, i)
                    for j in self._two_tdn._inter_edges:
                        parent_node_name = j[0].name.split("_", 1)[0] + "_" + str(i - 1)
                        child_node_name = j[1].name.split("_", 1)[0] + "_" + str(i)
                        """Connect copied nodes via inter edges like described in the two tdn"""
                        unrolled_net.add_edge(unrolled_net.node_lookup[parent_node_name],
                                              unrolled_net.node_lookup[child_node_name])
                    for j in self._two_tdn._intra_edges:
                        parent_node_name = j[0].name.split("_", 1)[0] + "_" + str(i)
                        child_node_name = j[1].name.split("_", 1)[0] + "_" + str(i)
                        """Connect copied nodes via inter edges like described in the two tdn"""
                        unrolled_net.add_edge(unrolled_net.node_lookup[parent_node_name],
                                              unrolled_net.node_lookup[child_node_name])
                        print "child"
                        print child_node_name
                        print unrolled_net.node_lookup[child_node_name].parents
                        # print self._d0.node_lookup[j[1].name.split("_", 1)[0] + "_" + str(0)].cpd
                        unrolled_net.node_lookup[child_node_name].set_cpd(
                            self._d0.node_lookup[j[1].name.split("_", 1)[0]
                                                 + "_" + str(0)].cpd)
                        print "copied"

                else:
                    unrolled_net.copy_nodes_indexed(t_plus_1_slice, i)
                    # in case we dont want the decisions to be copied for every new timeslice but only for every
                    # two. Write this and leave the decisions node out of the timeslices in both d0 and two_tbn network:
                    # unrolled_net.copy_nodes_indexed(self._two_tdn.decision_nodes, i - 1)

                    unrolled_net.copy_nodes_indexed(self._two_tdn.utility_nodes, i - 1)

                    for j in self._two_tdn._inter_edges:
                        parent_node_name = j[0].name.split("_", 1)[0] + "_" + str(i - 1)
                        child_node_name = j[1].name.split("_", 1)[0] + "_" + str(i)
                        unrolled_net.add_edge(unrolled_net.node_lookup[parent_node_name],
                                              unrolled_net.node_lookup[child_node_name])

                    for j in self._two_tdn._intra_edges:
                        parent_node_name = j[0].name.split("_", 1)[0] + "_" + str(i)
                        child_node_name = j[1].name.split("_", 1)[0] + "_" + str(i)
                        unrolled_net.add_edge(unrolled_net.node_lookup[parent_node_name],
                                              unrolled_net.node_lookup[child_node_name])
                        unrolled_net.node_lookup[child_node_name].set_cpd(
                            self._d0.node_lookup[j[1].name.split("_", 1)[0]
                                                 + "_" + str(0)].cpd)

            unrolled_net.set_partial_ordering_u(self._d0.get_partial_ordering(), length)
        else:
            raise Exception("Can't unroll over length: 0")
        return unrolled_net


class d0_net(object):

    def __init__(self):
        self.timeslice = {}
        self.utility_nodes = []
        self.decision_nodes = []
        self.random_nodes = []
        self.node_lookup = {}
        self.partialOrdering = []

    def set_zero_timeslice(self, nodes):
        self.timeslice[0] = nodes

    def add_nodes(self, nodes):
        for i in nodes:
            if isinstance(i, RandomNode):
                if i.name in self.node_lookup.keys():
                    pass
                if isinstance(i, DiscreteNode):
                    self.random_nodes.append(i)
                elif isinstance(i, UtilityNode):
                    self.utility_nodes.append(i)
                elif isinstance(i, DecisionNode):
                    self.decision_nodes.append(i)
                else:
                    raise Exception("Tried to add a node which the D0 Network can not work with")
                self.node_lookup[i.name] = i
            else:
                raise Exception("Can only add 'Node' and its subclasses as nodes into the D0 Network")

    def add_edge(self, node_from, node_to):

        node_to.add_parent(node_from)

    def get_partial_ordering(self):
        return self.partialOrdering

    def set_partial_ordering(self, partial_order):
        self.partialOrdering = partial_order

    def get_random_nodes(self):
        return self.random_nodes

    def get_decision_nodes(self):
        return self.decision_nodes

    def get_utility_nodes(self):
        return self.utility_nodes

    def get_all_nodes(self):
        return self.random_nodes + self.decision_nodes + self.utility_nodes

    def get_all_node_names(self):
        return self.node_lookup.keys()


class Two_TDN(object):
    def __init__(self, inter_edges=None, intra_edges=None):

        self.timeslice = {}
        self.utility_nodes = []
        self.decision_nodes = []
        self.random_nodes = []
        self.node_lookup = {}
        self._inter_edges = []
        self._intra_edges = []
        if inter_edges is not None:
            self.add_inter_edges(inter_edges)
        if intra_edges is not None:
            self.add_intra_edges(intra_edges)

    def add_nodes(self, nodes):
        for i in nodes:
            if isinstance(i, RandomNode):
                if i.name in self.node_lookup.keys():
                    pass
                if isinstance(i, DiscreteNode):
                    self.random_nodes.append(i)
                elif isinstance(i, UtilityNode):
                    self.utility_nodes.append(i)
                elif isinstance(i, DecisionNode):
                    self.decision_nodes.append(i)
                else:
                    raise Exception("Tried to add a node which the D0 Network can not work with")
                self.node_lookup[i.name] = i
            else:
                raise Exception("Can only add 'Node' and its subclasses as nodes into the D0 Network")

    def get_random_nodes(self):
        return self.random_nodes

    def get_decision_nodes(self):
        return self.decision_nodes

    def get_utility_nodes(self):
        return self.utility_nodes

    def get_all_nodes(self):
        return self.random_nodes + self.decision_nodes + self.utility_nodes

    def get_all_node_names(self):
        return self.node_lookup.keys()

    def add_inter_edge(self, node, node_p):
        """Add a transition connecting nodes when unrolling the network.

        The transition is a directed edge from node X_i (`node`) to node X_i'
        (`node_p`).

        Parameters
        ----------
        node : RandomNode, String
            The node X_i in the next time-slice.

        node_p: RandomNode, String
            The corresponding node X_i' in the current time-slice.

        Raises
        ------
        primo2.exceptions.StructureError
            If a node specified in the transition model cannot be found in the
            corresponding networks.
        """
        if node_p not in self.get_all_nodes():
            raise exceptions.StructureError(
                'Node "{}" is not found in D_{->} network "{}".')
        if node not in self.get_all_nodes():
            raise exceptions.StructureError(
                'Node "{}" is not found in D_{{->}} network "{}".')
        self._inter_edges.append((node, node_p))

    def add_inter_edges(self, inter_edges):
        """Add multiple transitions connecting nodes when unrolling the network.

        Parameters
        ----------
        inter_edges : [(node, node_p),]
            A list of pairs, each of which represents one transition.
            See add_transition for more information.
        """
        for inter_edges in inter_edges:
            self.add_inter_edge(inter_edges[0], inter_edges[1])

    def add_intra_edge(self, node, node_p):
        """Add a transition connecting nodes when unrolling the network.

        The transition is a directed edge from node X_i (`node`) to node X_i'
        (`node_p`).

        Parameters
        ----------
        node : RandomNode, String
            The node X_i in the next time-slice.

        node_p: RandomNode, String
            The corresponding node X_i' in the current time-slice.

        Raises
        ------
        primo2.exceptions.StructureError
            If a node specified in the transition model cannot be found in the
            corresponding networks.
        """
        if node_p not in self.get_all_nodes():
            raise exceptions.StructureError(
                'Node "{}" is not found in D_{->} network "{}".')
        if node not in self.get_all_nodes():
            raise exceptions.StructureError(
                'Node "{}" is not found in D_{{->}} network "{}".')
        self._intra_edges.append((node, node_p))

    def add_intra_edges(self, intra_edges):
        """Add multiple transitions connecting nodes when unrolling the network.

        Parameters
        ----------
        intra_edges : [(node, node_p),]
            A list of pairs, each of which represents one transition.
            See add_transition for more information.
        """
        for intra_edges in intra_edges:
            self.add_intra_edge(intra_edges[0], intra_edges[1])

    def set_t_timeslice(self, initial_nodes):
        self.timeslice['t'] = initial_nodes

    def set_t_plus_one_timeslice(self, next_nodes):
        self.timeslice['t_plus_1'] = next_nodes


class DecisionNetwork(object):

    def __init__(self):
        self.node_lookup = {}
        self.partialOrdering = []
        self.random_nodes = []
        self.decision_nodes = []
        self.utility_nodes = []

    def add_node(self, node):
        if isinstance(node, RandomNode):
            if node.name in self.node_lookup.keys():
                pass
            if isinstance(node, DiscreteNode):
                self.random_nodes.append(node)
            elif isinstance(node, UtilityNode):
                self.utility_nodes.append(node)
            elif isinstance(node, DecisionNode):
                self.decision_nodes.append(node)
            else:
                raise Exception("Tried to add a node which the Bayesian Decision Network can not work with")
            self.node_lookup[node.name] = node
        else:
            raise Exception("Can only add 'Node' and its subclasses as nodes into the BayesianNetwork")

    def add_nodes(self, nodes):
        for i in nodes:
            if isinstance(i, RandomNode):
                if i.name in self.node_lookup.keys():
                    pass
                if isinstance(i, DiscreteNode):
                    self.random_nodes.append(i)
                elif isinstance(i, UtilityNode):
                    self.utility_nodes.append(i)
                elif isinstance(i, DecisionNode):
                    self.decision_nodes.append(i)
                else:
                    raise Exception("Tried to add a node which the Bayesian Decision Network can not work with")
                self.node_lookup[i.name] = i
            else:
                raise Exception("Can only add 'Node' and its subclasses as nodes into the BayesianNetwork")

    def copy_nodes_indexed(self, nodes_to_copy, index):
        for i in nodes_to_copy:
            if i.name.split("_", 1)[0] + "_" + str(index) not in self.get_all_node_names():
                if isinstance(i, RandomNode):
                    if isinstance(i, DiscreteNode):
                        new_node = DiscreteNode(i.name.split("_", 1)[0] + "_" + str(index), i.values)
                        self.random_nodes.append(new_node)
                        self.node_lookup[new_node.name] = new_node
                    elif isinstance(i, UtilityNode):
                        new_node = UtilityNode(i.name.split("_", 1)[0] + "_" + str(index))
                        self.utility_nodes.append(new_node)
                        self.node_lookup[new_node.name] = new_node

                    elif isinstance(i, DecisionNode):
                        new_node = DiscreteNode(i.name.split("_", 1)[0] + "_" + str(index), i.values)
                        self.decision_nodes.append(new_node)
                        self.node_lookup[new_node.name] = new_node
                    else:
                        raise Exception("Tried to copy a node which the Bayesian Decision Network can not work with")

                else:
                    raise Exception("Can only add 'Node' and its subclasses as nodes into the BayesianNetwork")
            else:
                raise Exception("A copy of this node already exists")

    def add_edge(self, node_from, node_to):

        self.node_lookup[node_to].add_parent(self.node_lookup[node_from])

    def get_partial_ordering(self):
        return self.partialOrdering

    def set_partial_ordering(self, partial_order):
        self.partialOrdering = partial_order

    def extend_partial_ordering(self, extension):
        self.partialOrdering.append(extension)

    def set_partial_ordering_u(self, init_order, length):

        for i in range(0, length):
            for j in init_order:
                # print(type(j))
                if isinstance(j, list):
                    if isinstance(self.node_lookup[j[0]], DiscreteNode) or \
                            isinstance(self.node_lookup[j[0]], DecisionNode):
                        temp = [value.split("_", 1)[0] + "_" + str(i) for value in j]
                        self.partialOrdering.append(temp)
                    # Write this if decisions are over 2 timeslices
                    # elif i < length - 1 or i == 0 or decisions:
                    #     temp = [value.split("_", 1)[0] + "_" + str(i) for value in j]
                    #     self.partialOrdering.append(temp)
                elif isinstance(j, str):
                    # print j
                    # print(self.node_lookup)
                    if isinstance(self.node_lookup[j], DiscreteNode) or \
                            isinstance(self.node_lookup[j], DecisionNode):
                        self.partialOrdering.append(j.split("_", 1)[0] + "_" + str(i))
                    # Write this if decisions are over 2 timeslices
                    # elif i < length - 1 or i == 0:
                    #     self.partialOrdering.append(j.split("_", 1)[0] + "_" + str(i))

    def get_random_nodes(self):
        '''Returns all RandomNodes'''
        return self.random_nodes

    def get_decision_nodes(self):
        return self.decision_nodes

    def get_utility_nodes(self):
        return self.utility_nodes

    def get_all_nodes(self):
        return self.random_nodes + self.decision_nodes + self.utility_nodes

    def get_all_node_names(self):
        return self.node_lookup.keys()
