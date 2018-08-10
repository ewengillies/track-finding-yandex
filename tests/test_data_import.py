"""
Tests for importing data from root file
"""
from __future__ import print_function
import sys
import pytest
import numpy as np
sys.path.insert(0, "../modules")
from root_numpy import list_branches
import hits

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
# test files we will use
FILES = ["test_file_a"]
# names used for import that belong together
NAMES = {}
NAMES["CDC"] = ("CDCHitTree", "CDCHit.f")
NAMES["CTH"] = ("CTHHitTree", "CTHHit.f")
# some specific test branches for when they are needed
BRANCHES = ['Track.fStartMomentum.fX',
            'Track.fStartMomentum.fY',
            'Track.fStartMomentum.fZ']

# TODO 
# test empty branch import
# test selection import
# test get event
# test get measurement
# test sort hits

# HELPER FUNCTIONS #############################################################
def filter_branches(branches):
    """
    Filter out the branches that are automatically imported
    """
    ret_val = branches
    for filter_br in ["CDCHit.fDetectedTime",
                      "CDCHit.fCharge",
                      "CDCHit.fEventNumber",
                      "CDCHit.fIsSig",
                      'CTHHit.fMCPos.fE',
                      'CTHHit.fCharge',
                      'CTHHit.fEventNumber',
                      'CTHHit.fIsSig']:
        ret_val = [brch for brch in ret_val if filter_br not in brch]
    return ret_val

# TEST BASIC IMPORT FUNCTIONS ##################################################

@pytest.fixture(params=[
    # Parameterize the array construction
    # path, geom, branches, 
    (FILES[0], "CDC", 'all'),
    (FILES[0], "CTH", 'all'),
    (FILES[0], "CDC", [NAMES["CDC"][1]+branch for branch in BRANCHES]),
    (FILES[0], "CTH", [NAMES["CTH"][1]+branch for branch in BRANCHES])])
def cstrct_hits_params(request):
    """
    Parameterize the flat hit parameters
    """
    return request.param

@pytest.fixture()
def flat_hits(cstrct_hits_params):
    """
    Construct the base flat hits object
    """
    # Unpack the parameters
    file, geom, branch = cstrct_hits_params
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    if branch == "all":
        branch = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample = hits.FlatHits(root_file,
                           tree=tree,
                           prefix=prefix,
                           branches=branch)
    return sample, file, geom

@pytest.fixture()
def flat_hits_and_ref(flat_hits):
    """
    Package the hits and the reference data together
    """
    # Unpack the parameters
    sample, file, geom = flat_hits
    # Check that it is the same as the first time we loaded in this data
    reference_file = file+"_"+geom+".npz"
    reference_data = np.load(reference_file)["array"]
    # Return the information
    return sample, reference_data

def test_sample_columns(flat_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data = flat_hits_and_ref
    # Ensure column names are subset of reference names
    ref_columns = list(reference_data.dtype.names)
    ref_columns.sort()
    new_columns = list(sample.data.dtype.names)
    new_columns.sort()
    miss_cols = list(set(new_columns) - set(ref_columns))
    assert not miss_cols, "Columns in loaded sample are not found in "+\
        "reference sample \n{}".format("\n".join(miss_cols))

def test_sample_data(flat_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data = flat_hits_and_ref
    # Ensure all the data is the same
    for col in sample.data.dtype.names:
        new_data = sample.data[col]
        ref_data = reference_data[col]
        np.testing.assert_allclose(new_data, ref_data)

# FLAT HITS SELECTED TESTS #####################################################
@pytest.fixture(params=[
    # Parameterize the array construction
    # selection
    [BRANCHES[0] + " < 0"],
    [BRANCHES[1] + " > 0"],
    [BRANCHES[1] + " > 0", BRANCHES[0] +" < 0"],
    [BRANCHES[1] + " > 0", BRANCHES[0] +" < 0", BRANCHES[2] + " > 0"],
    ["EventNumber == 1", BRANCHES[0] +" < 0"]
    ])
def cstrct_hits_params_sel(request, cstrct_hits_params):
    """
    Parameterize the flat hit parameters
    """
    file, geom, branch = cstrct_hits_params
    selection = request.param
    selection = " && ".join(NAMES[geom][1]+sel for sel in selection)
    return file, geom, branch, selection

@pytest.fixture()
def flat_hits_sel(cstrct_hits_params_sel):
    """
    Construct the base flat hits object with some selections
    """
    # Unpack the parameters
    file, geom, branch, selection = cstrct_hits_params_sel
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    if branch == "all":
        branch = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample = hits.FlatHits(root_file,
                           tree=tree,
                           prefix=prefix,
                           selection=selection,
                           branches=branch)
    return sample, selection

def test_flat_hits_sel(flat_hits_sel):
    """
    Test that the selections worked
    """
    sample, selection = flat_hits_sel
    # Deconstruct the selections
    sel_list = selection.split(" && ")
    sel_list = [sel.split(" ") for sel in sel_list]
    error_message = "All values must pass the relations: {}".format(selection)
    for branch, relation, value in sel_list:
        if relation == "<":
            print("Failed {} {} {}".format(branch, relation, value))
            assert np.all(sample.data[branch] < float(value)), error_message
        elif relation == ">":
            print("Failed {} {} {}".format(branch, relation, value))
            assert np.all(sample.data[branch] > float(value)), error_message
        elif relation == "==":
            print("Failed {} {} {}".format(branch, relation, value))
            assert np.all(sample.data[branch] == float(value)), error_message
        else:
            error_message = "Relation symbol {} not supported".format(relation)
            raise AssertionError(error_message)
