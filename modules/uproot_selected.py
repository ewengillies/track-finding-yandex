"""
Tools to help with low-memory, selected uproot imports
"""
from __future__ import print_function
import re
import numpy as np
import uproot
import pandas as pd

def _parse_nested_parens(text, left=r'[(]', right=r'[)]', sep=r','):
    """
    This parses nested parentheses into nested lists.  Its lifted from
    StackOverload.
    Based on https://stackoverflow.com/a/17141899/190597 (falsetru)
    """
    pat = r'({}|{}|{})'.format(left, right, sep)
    tokens = re.split(pat, text)
    stack = [[]]
    for tlk in tokens:
        if not tlk or re.match(sep, tlk):
            continue
        if re.match(left, tlk):
            stack[-1].append([])
            stack.append(stack[-1][-1])
        elif re.match(right, tlk):
            stack.pop()
            if not stack:
                raise ValueError('error: opening bracket is missing')
        else:
            stack[-1].append(tlk)
    if len(stack) > 1:
        print(stack)
        raise ValueError('error: closing bracket is missing')
    return stack.pop()

def _parse_operations(parsed_parenths, oper_dict, join_dict):
    """
    This parses each comparison operation string in the nested lists
    into a (function, branch, value) tuple.

    ex: [["branch_a < 10", "&&", "branch_b > 5"], || "branch_c == 5"] ->
        [[(np.greater, "branch_a, float(10)), "&&",
          (np.less,    "branch_b, float(5))], "||",
         (np.equal, "branch_c", 5)]
    :param: parse_parenths, list
        Nested list of lists
    :param: oper_dict, dict
        Mapping from operation as string to corresponding np.func
    :param: join_dict, dict
        Mapping from joining function to corresponding logical np.func
    """
    # Check if its a list, if so, recurse until its a string
    return_val = []
    for expression in parsed_parenths:
        if not isinstance(expression, str):
            return_val += [_parse_operations(expression, oper_dict, join_dict)]
        else:
            # Split the input on whitespace
            compare = expression.split()
            # Iterate through
            for index, expr in enumerate(compare):
                # Check if our current string is an expression
                if expr in oper_dict.keys():
                    return_val += [(oper_dict[expr],
                                    compare[index-1],
                                    float(compare[index+1]))]
                # If its a logical comparison, save it for later
                elif expr in join_dict.keys():
                    return_val += [expr]
    # Return the list
    return return_val

def _parse_connections(parsed_opts, join_dict):
    """
    This parses all the connection operations, and replaces the list nesting
    with tuple nesting.

    ex: [[(np.greater, "branch_a, float(10)), "&&",
          (np.less,    "branch_b, float(5))], "||",
         (np.equal, "branch_c", 5)] ->
        (np.logical_or,
         (np.logical_and,
          (np.greater, 'branch_a', 10.0),
          (np.less, 'branch_b', 5.0)),
         (np.equal, 'branch_c', 5.0))

    :param: parse_opts, list
        Nested lists of operations and joining operation strings
    :param: join_dict, dict
        Mapping from joining function to corresponding logical np.func
    """
    # Check if its a list
    if not isinstance(parsed_opts, list):
        return parsed_opts
    return_val = _parse_connections(parsed_opts[0], join_dict)
    for index, expr in enumerate(parsed_opts):
        # Check if our current string is an expression
        if isinstance(expr, str) and (expr in join_dict.keys()):
            # If it is, parse the next operation and join it to the current one
            next_opt = _parse_connections(parsed_opts[index+1], join_dict)
            # Nest a tuple of tuples
            return_val = (join_dict[expr], return_val, next_opt)
    # Return this new stack of operations
    return return_val

def _parse_selection(selection_string):
    """
    This parses a selection string that worked for root_numpy selection
    parameter into nested tuples of operation, branch names, and values. It
    assumes white space seperation between between branch names, comparison
    operations (<,>,==,!=, etc.), joining operations (&&, ||), and values.
    Parentheses need no white spaces.

    ex: "(branch_a < 10 && branch_b < 5) || branch_c == 5"->
        (np.logical_or,
         (np.logical_and,
          (np.greater, 'branch_a', 10.0),
          (np.less, 'branch_b', 5.0)),
         (np.equal, 'branch_c', 5.0))

    :param: selection string
        Selection string that could be passed into root_numpy
    """
    # Define all valid operations
    oper_dict = {"==" : np.equal,
                 "!=" : np.not_equal,
                 ">=" : np.greater_equal,
                 "<=" : np.less_equal,
                 ">"  : np.greater,
                 "<"  : np.less}
    # Define all the connection operations we can do
    join_dict = {"&&" : np.logical_and,
                 "||" : np.logical_or}
    # Start by dealing with parentheses
    parsed_parenths = _parse_nested_parens(selection_string)
    # Parse the operations into tuples
    pars_operations = _parse_operations(parsed_parenths, oper_dict, join_dict)
    # Parse out the connections
    return _parse_connections(pars_operations, join_dict)

def eval_selection(selection_string, data):
    """
    Evaluate the selection string over the data, similar to root_numpy selection
    parameter. This selection string must have at least one white space between
    branch names, comparison operations, and values, eg:

    "(branch_a < 10 && branch_b < 5) || branch_c == 5" is valid

    "(branch_a<10&&branch_b<5)||branch_c==5" is not valid.

    No white space is needed around the parentheses.  Branch names must come
    first in the comparison, i.e. (branch_a < 5) not (5 > branch_a).

    Finally, the data must be able to be indexed by branch name,
    i.e. data[branch_name] needs to return something that will work inside
    np.ufuncs. Pandas DataFrame or numpy structured arrays should work.

    This function returns a boolean mask for numpy structured arrays, and
    a boolean series for dataframes.

    :param: selection_string, str
        A string to define what selections will be made
    :param: data, dataframe or numpy.record
        Data whose columns can be indexed by a string
    """
    # Evaluate the first level
    first_level = _parse_selection(selection_string)
    # Recusrse on this selection
    return _recurse_eval(first_level, data)

def _recurse_eval(this_level, data):
    opp, val_a, val_b = this_level
    if isinstance(val_b, float):
        # Cast to the correct type using a numpy hack
        cast_b = np.asscalar(np.array([val_b], dtype=data[val_a].dtype))
        # Return the value of the operation
        return opp(data[val_a], cast_b)
    # Otherwise, opp is either && or || and we must dig deeper
    return opp(_recurse_eval(val_a, data),
               _recurse_eval(val_b, data))

def get_selection_branches(selection_string):
    """
    Figure out which branches are being used in the selection so that we can be
    sure to import them

    :param: selection_string, str
        A string to define what selections will be made
    :param: data, dataframe or numpy.record
        Data whose columns can be indexed by a string
    """
    first_level = _parse_selection(selection_string)
    return_list = []
    _recurse_selections(first_level, return_list)
    return list(set(return_list))

def _recurse_selections(next_level, return_list):
    # If we've found a float, then we've reach a "branch level"
    _, val_a, val_b = next_level
    if isinstance(val_b, float):
        # Cast to the correct type using a numpy hack
        return_list += [val_a]
    else:
        # Otherwise, opp is either && or || and we must dig deeper
        _recurse_selections(val_a, return_list)
        _recurse_selections(val_b, return_list)

def format_data(data, branches, selection=None, single_perc=True):
    """
    Format the block of data
    """
    # TODO docs
    # Rename the columns before we do the selection
    data.columns = [str(c)[2:-1] for c in data.columns.values]
    # Select out the variables
    if selection is not None:
        # Run the selection, then trim the sample back just have the
        # desired branches
        data = data[eval_selection(selection, data)][branches]
    # Convert to single percision if needed
    if single_perc:
        for brch, b_dtype in data.dtypes.items():
            if b_dtype == np.float64:
                data[brch] = data[brch].values.astype(np.float32)
    # Return the data
    return data

def import_uproot_selected(path, 
                           tree=None,
                           branches=None,
                           selection=None, 
                           num_entries=None,
                           single_perc=True):
    # TODO docs
    # Initialize the return value to none
    data = None
    # Import the branches
    file = uproot.open(path)
    # Check if we've specified a tree, otherwise take the first tree
    if tree is None:
        tree = file.keys()[0]
    file = file[tree]
    # Check if the branches of the tree are specified, if not, take all branches
    if branches is None:
        branches = list_branches(path, tree)
    # Get the branches needed for selection as well
    imprt_brchs = list(branches)
    if selection is not None:
        # Import branches should be these branches plus the ones we need to
        # select on
        sel_branches = get_selection_branches(selection)
        imprt_brchs += sel_branches
    # Ensure each branch is only imported once
    imprt_brchs = list(set(imprt_brchs))
    # If we are getting all the data at once, just use one call
    if num_entries is None:
        data = file.arrays(branches=imprt_brchs, outputtype=pd.DataFrame)
        data = format_data(data, branches,
                           selection=selection,
                           single_perc=single_perc)
    # If we are block importing do so
    else:
        data = []
        for d_blk in file.iterate(branches=imprt_brchs,
                                  outputtype=pd.DataFrame,
                                  entrysteps=num_entries):
            # Format the data block
            data = format_data(data, branches,
                               selection=selection,
                               single_perc=single_perc)
            # Add this data to our import
            data += [d_blk]
        # Concatenate the data
        data = pd.concat(data)
    return data

def check_for_branches(path, tree, branches, soft_check=False, verbose=False):
    """
    This checks for the needed branches before they are imported to avoid
    the program to hang without any error messages

    :param path: path to root file
    :param tree: name of tree in root file
    :param branches: required branches
    """
    # Get the names of the availible branches
    availible_branches = list_branches(path, tree)
    # Get the requested branches that are not availible
    bad_branches = list(set(branches) - set(availible_branches))
    # Otherwise, shut it down if its the wrong length
    if bad_branches:
        err_msg = "ERROR: The requested branches:\n"+\
                  "\n".join(bad_branches) + "\n are not availible\n"+\
                  "The branches availible are:\n"+"\n".join(availible_branches)
        if soft_check:
            if verbose:
                print(err_msg)
            return False
        # Check that this is zero in length
        assert not bad_branches, err_msg
        # Otherwise return true
    return True

def list_branches(path, tree=None):
    """
    List the branches availible
    """
    # TODO docs
    # Get the names of the availible branches
    availible_branches = uproot.open(path)
    # Check if we've specified a tree, otherwise take the first tree
    if tree is None:
        tree = availible_branches.keys()[0]
    availible_branches = availible_branches[tree]
    availible_branches = [str(c)[2:-1] for c in availible_branches.keys()]
    return availible_branches
