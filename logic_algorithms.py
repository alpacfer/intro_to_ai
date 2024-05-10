# logic_algorithms.py

import sympy
from sympy.logic.boolalg import Implies, And, Or, Not, Equivalent
from itertools import chain, combinations
from sympy.logic.inference import satisfiable

def contraction(plausibility_table, proposition):
    """
    Perform contraction with the given proposition.

    Parameters:
    plausibility_table (dict): The table with plausibility orders.
    proposition (str): The proposition to contract with (e.g., 'p', 'q', 'n_p', 'n_q').

    Returns:
    tuple: The most plausible state where the proposition is true and the most plausible state where the proposition is false.
    """
    # Find the most plausible state where the proposition is true
    most_plausible_true_state = None
    for key, value in sorted(plausibility_table.items(), key=lambda item: item[1]['order']):
        if proposition in key.split(', '):
            most_plausible_true_state = value['states']
            break

    # Find the most plausible state where the proposition is false
    most_plausible_false_state = None
    for key, value in sorted(plausibility_table.items(), key=lambda item: item[1]['order']):
        if proposition not in key.split(', '):
            most_plausible_false_state = value['states']
            break

    return (most_plausible_true_state, most_plausible_false_state)

def revision(plausibility_table, proposition):
    """
    Perform revision with the given proposition.

    Parameters:
    plausibility_table (dict): The table with plausibility orders.
    proposition (str): The proposition to revise with (e.g., 'p', 'q', 'n_p', 'n_q').

    Returns:
    list: The most plausible state where the proposition is true.
    """
    # Find the most plausible state where the proposition is true
    most_plausible_state = None
    for key, value in sorted(plausibility_table.items(), key=lambda item: item[1]['order']):
        if proposition in key.split(', '):
            most_plausible_state = value['states']
            break

    return most_plausible_state

def all_subsets(s):
    """Generate all subsets of a set."""
    return list(chain(*map(lambda x: combinations(s, x), range(0, len(s)+1))))

def implies_q(subset, q):
    """Check if a subset of A implies q."""
    # Convert subset to a conjunction of its elements
    conjunction = True
    for expr in subset:
        conjunction = conjunction & expr
    
    # Check if conjunction implies q
    return not satisfiable(conjunction & Not(q))

def is_subset(subset1, subset2):
    """Check if subset1 is a subset of subset2."""
    return set(subset1).issubset(set(subset2))

def maximal_subsets(subsets):
    """Find the maximal subsets from a list of subsets."""
    maximal = []
    for subset in subsets:
        if not any(is_subset(subset, other) for other in subsets if subset != other):
            maximal.append(subset)
    return maximal

def bb_contraction(belief_base, proposition):
    """Find the maximal subsets of belief_base that do not imply proposition."""
    subsets_A = all_subsets(belief_base)
    candidate_subsets = [subset for subset in subsets_A if not implies_q(subset, proposition)]
    A_contracted_q = maximal_subsets(candidate_subsets)
    return A_contracted_q