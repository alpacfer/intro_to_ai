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




def bb_contraction(belief_base, proposition):
    """Find the maximal subsets of belief_base that do not imply the proposition."""
    subsets_A = all_subsets(belief_base)
    candidate_subsets = [subset for subset in subsets_A if not implies_q(subset, proposition)]
    A_contracted_q = maximal_subsets(candidate_subsets)
    return A_contracted_q

def all_subsets(s):
    """Return all subsets of a set."""
    return list(chain(*[combinations(s, r) for r in range(len(s) + 1)]))

def implies_q(subset, proposition):
    """Check if a subset implies a given proposition."""
    conjunction = sp.And(*subset)
    return sp.Implies(conjunction, proposition).simplify() == True

def maximal_subsets(subsets):
    """Return the maximal subsets from a list of subsets."""
    maximal = []
    for subset in subsets:
        if not any(set(subset).issubset(set(other)) for other in subsets if subset != other):
            maximal.append(subset)
    return maximal

def is_consistent(belief_base):
    """Check if a belief base is consistent."""
    conjunction = sp.And(*belief_base)
    return bool(satisfiable(conjunction))





def contract_belief_base(belief_base, new_belief):
    """Find the maximal subsets of belief_base that do not imply the negation of new_belief."""
    all_subsets_base = get_all_subsets(belief_base)
    valid_subsets = [subset for subset in all_subsets_base if not subset_implies(subset, Not(new_belief))]
    contracted_belief_base = get_maximal_subsets(valid_subsets)
    return contracted_belief_base

def get_all_subsets(s):
    """Return all subsets of a set."""
    return list(chain(*[combinations(s, r) for r in range(len(s) + 1)]))

def subset_implies(subset, proposition):
    """Check if a subset implies a given proposition."""
    conjunction = sp.And(*subset)
    return sp.Implies(conjunction, proposition).simplify() == True

def get_maximal_subsets(subsets):
    """Return the maximal subsets from a list of subsets."""
    maximal = []
    for subset in subsets:
        if not any(set(subset).issubset(set(other)) for other in subsets if subset != other):
            maximal.append(subset)
    return maximal

def check_consistency(belief_base):
    """Check if a belief base is consistent."""
    conjunction = sp.And(*belief_base)
    return bool(satisfiable(conjunction))




















def convert_to_cnf(expressions):
    """
    Convert a list of logical expressions to Conjunctive Normal Form (CNF).

    Args:
    expressions (list of sympy.Expr): A list of sympy logical expressions.

    Returns:
    list of sympy.Expr: A list of CNF expressions corresponding to the input expressions.
    """
    return [to_cnf(expr, simplify=True) for expr in expressions]

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

def display_plausibility_table(plausibility_table):
    # Extract the columns and rows from the plausibility table
    columns = ['p, q', 'p, q̅', 'p̅, q', 'p̅, q̅']
    max_order = max(item['order'] for item in plausibility_table.values())
    rows = list(range(max_order))
    
    # Initialize an empty DataFrame
    data = {col: [''] * max_order for col in columns}
    
    # Fill in the DataFrame with the states based on the plausibility order
    for key, value in plausibility_table.items():
        col_name = key.replace('n_q', 'q̅').replace('n_p', 'p̅')
        state = value['states'][0]
        row_index = max_order - value['order']  # Invert order to have the most plausible at the bottom
        data[col_name][row_index] = state
    
    df = pd.DataFrame(data)
    
    # Apply vertical lines to the DataFrame using Styler
    styled_df = df.style.set_table_styles(
        [{'selector': 'td, th',
          'props': [('border', '1px solid black')]}]
    )
    
    return styled_df