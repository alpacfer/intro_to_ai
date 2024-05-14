# logic_algorithms.py

import sympy
from sympy.logic.boolalg import Implies, And, Or, Not, Equivalent
from itertools import chain, combinations
from sympy.logic.inference import satisfiable
import sympy as sp
from sympy.logic.boolalg import to_cnf
from sympy.logic.boolalg import And, Not, Implies
from itertools import product
import pandas as pd

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
    return not satisfiable(conjunction & sp.Not(q))

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

# Function to check consistency
def is_consistent(belief_base):
    """Check if a belief base is consistent."""
    conjunction = True
    for belief in belief_base:
        conjunction = conjunction & belief
    return satisfiable(conjunction)

def convert_to_cnf(expression):
    """
    Convert a logical expression to Conjunctive Normal Form (CNF).

    Args:
    expression (sympy.Expr): A sympy logical expression.

    Returns:
    sympy.Expr: The CNF of the input expression.
    """
    return to_cnf(expression, simplify=True)

# Function to convert boolean values to 'T' or 'F'
def bool_to_tf(value):
    return 'T' if value else 'F'

# Create a function to compute the truth table
def compute_truth_table(variables, formulas):
    # Generate all combinations of truth values for the given variables
    combinations = list(product([False, True], repeat=len(variables)))
    
    # Prepare the header
    header = [str(var) for var in variables] + [str(formula) for formula in formulas]
    
    # Prepare the rows of the table
    rows = []
    for combination in combinations:
        assignments = dict(zip(variables, combination))
        row = [bool_to_tf(value) for value in combination]
        for formula in formulas:
            row.append(bool_to_tf(formula.subs(assignments)))
        rows.append(row)
    
    # Create a DataFrame
    df = pd.DataFrame(rows, columns=header)
    
    # Define a function to apply styles
    def highlight(cell):
        if cell == 'T':
            return 'background-color: lightgreen; color: white'
        elif cell == 'F':
            return 'background-color: lightcoral; color: white'
        return ''
    
    # Apply the styles and display the DataFrame
    styled_df = df.style.applymap(highlight)
    return styled_df

def compute_belief_revision_table(formulas, belief_revision):
    # Define the symbols
    p, q = sp.symbols('p q')

    # Generate all combinations of truth values for p and q
    combinations = list(product([False, True], repeat=2))

    # Create a list to store the rows
    rows = []

    # Evaluate each combination
    for combination in combinations:
        p_val, q_val = combination
        row = {
            'p': p_val,
            'q': q_val,
            'Belief Revision: ¬p': belief_revision.subs({p: p_val, q: q_val})
        }
        
        # Evaluate the original formulas
        for formula, expr in formulas.items():
            row[formula] = expr.subs({p: p_val, q: q_val})
        
        # Evaluate the formulas after applying belief revision (p = False)
        revised_values = {p: False, q: q_val}
        for formula, expr in formulas.items():
            revised_formula = f'After ¬p: {formula}'
            row[revised_formula] = expr.subs(revised_values)
        
        rows.append(row)

    # Create a DataFrame from the rows
    table = pd.DataFrame(rows)

    # Display the table
    def bool_to_tf(value):
        return 'T' if value else 'F'

    styled_table = table.style.applymap(lambda val: 'background-color: lightgreen; color: white' if val else 'background-color: lightcoral; color: white').format(bool_to_tf)
    return styled_table