"""
Tests for importing data from root file
"""
from __future__ import print_function
import sys
import pytest
import numpy as np
from numpy.testing import assert_allclose
sys.path.insert(0, "../modules")
from root_numpy import list_branches
import hits
from hits import check_for_branches, _add_name_to_branches

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
# test files we will use
FILES = ["ref_data/test_file_a"]
# names used for import that belong together
NAMES = {}
NAMES["CDC"] = ("CDCHitTree", "CDCHit.f")
NAMES["CTH"] = ("CTHHitTree", "CTHHit.f")
# some specific test branches for when they are needed
BRANCHES = ['Track.fStartMomentum.fX',
            'Track.fStartMomentum.fY',
            'Track.fStartMomentum.fZ',
            'MCPos.fP.fX',
            'MCPos.fP.fY',
            'MCPos.fP.fZ']
# Turn this on if we want to regenerate the reference sample
GENERATE_REFERENCE = False
# Number of branches expected for reference samples
N_BRANCHES = {}
N_BRANCHES["CDC"] = 34
N_BRANCHES["CTH"] = 32


# HELPER FUNCTIONS #############################################################

def generate_reference(reference_file, sample, n_branches, desired_branches):
    # Generate the largest sample of reference data possible
    if n_branches == desired_branches:
        # Try to convert it to records if its a pandas DF
        try:
            np.savez(reference_file, array=sample.to_records(index=False))
        # Otherwise assume its already a numpy array
        except AttributeError:
            np.savez(reference_file, array=sample)
        #raise AssertionError("Generated new reference, test is tautological")

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

def check_columns(sample, reference_data):
    """
    Helper function to ensure all columns are imported
    """
    ref_columns = sorted(list(reference_data.dtype.names))
    new_columns = sorted(list(sample.data.columns.values))
    miss_cols = list(set(new_columns) - set(ref_columns))
    # TODO remove this index checker
    assert not miss_cols, "Columns in loaded sample are not found in "+\
        "reference sample \n{}".format("\n".join(miss_cols))

def check_data(sample, reference_data):
    """
    Helper function to ensure all columns are imported
    """
    for col in sample.data.columns.values:
        new_data = sample.data[col].values
        print("Checking column:", col)
        ref_data = reference_data[col]
        assert_allclose(new_data, ref_data, err_msg=col)

def check_filter(filtered_hits, variable, values, greater, less, invert):
    """
    Helper function for filter tests
    """
    # Ensure there are some events left
    assert len(filtered_hits) != 0, "Filter has removed all hits!"+\
            " This has resulted in a trivial check"
    # Ensure they pass all the filter requirements
    err = "Sample filtered on variable {} ".format(variable)
    if values is not None:
        test = np.all(np.in1d(filtered_hits[variable], values))
        if invert:
            test = not test
        assert test, \
            err + "did not filter by values {} correctly".format(values)+\
            " {}".format(filtered_hits[variable])
    if greater is not None:
        test = np.all(filtered_hits[variable] > greater)
        if invert:
            test = not test
        assert test, \
            err + "did not filter by 'greater than' {} correctly".format(greater)+\
            " {}".format(filtered_hits[variable])
    if less is not None:
        test = np.all(filtered_hits[variable] < less)
        if invert:
            test = not test
        assert test, \
            err + "did not filter by 'greater than' {} correctly".format(less)+\
            " {}".format(filtered_hits[variable])

# TEST HELPER FUNCTIONS ########################################################

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
def good_bad_branches(cstrct_hits_params):
    """
    Build some good and bad branch names
    """
    # Unpack the parameters
    file, geom, branches = cstrct_hits_params
    tree, _ = NAMES[geom]
    a_file = file + ".root"
    # Load all the branches
    if branches == "all":
        branches = list_branches(a_file, treename=tree)
    # Return the list of good and bad branches
    bad_branches = ["garbage" + brch for brch in branches]
    return branches, bad_branches, a_file, tree

def test_check_branch(good_bad_branches):
    """
    Check if testing for the branch in the data works
    """
    # Unpack the arguments
    branches, bad_branches, a_file, tree = good_bad_branches
    # Check if we can find the branch(es) we expect
    err_msg = "Did not find expected branches:\n{}".format("\n".join(branches))
    found_good = check_for_branches(a_file, tree, branches, soft_check=True)
    assert found_good, err_msg
    # Check that bad equests are rejected
    err_msg = "Found non-existant branches:\n{}".format("\n".join(bad_branches))
    found_bad = check_for_branches(a_file, tree, bad_branches, soft_check=True)
    assert not found_bad, err_msg

def test_add_branch(good_bad_branches):
    """
    Check if testing for the branch in the data works
    """
    # Unpack the arguments
    branches, bad_branches, a_file, tree = good_bad_branches
    # Check if we can find the branch(es) we expect
    full_branches = None
    empty_branches = None
    for branch in branches + bad_branches:
        full_branches, empty_branches = _add_name_to_branches(a_file,
                                                              tree,
                                                              branch,
                                                              full_branches,
                                                              empty_branches)
    assert full_branches == branches,\
        "Existing branches not returned correctly\n{}".format("\n".join(branches))
    assert empty_branches == bad_branches,\
        "Empty branches not returned correctly\n{}".format("\n".join(bad_branches))

# TEST CONSTRUCTOR AND IMPORT ##################################################

@pytest.fixture()
def flat_hits(cstrct_hits_params):
    """
    Construct the base flat hits object
    """
    # Unpack the parameters
    file, geom, rqst_branches = cstrct_hits_params
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    branches = rqst_branches
    if branches == "all":
        branches = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample = hits.FlatHits(root_file,
                           tree=tree,
                           prefix=prefix,
                           branches=branches)
    # Assign every 5th hit as signal
    sample.data.loc[::5, sample.hit_type_name] = bool(True)
    return sample, file, geom, rqst_branches

@pytest.fixture()
def flat_hits_and_ref(flat_hits):
    """
    Package the hits and the reference data together
    """
    # Unpack the parameters
    sample, file, geom, rqst_branches = flat_hits
    # Check that it is the same as the first time we loaded in this data
    reference_file = file+"_"+geom+".npz"
    # Generate the referece file if needed
    if GENERATE_REFERENCE:
        n_branches = len(sample.data.columns.values)
        generate_reference(reference_file,
                           sample.data,
                           n_branches,
                           N_BRANCHES[geom])
    reference_data = np.load(reference_file)["array"]
    # Return the information
    return sample, reference_data, rqst_branches

def test_all_branches_present(flat_hits_and_ref):
    """
    Ensure we did not drop any branches unintentionally
    """
    # Unpack the values
    sample, reference_data, rqst_branches = flat_hits_and_ref
    # Ensure we have the right number of branches if we requested all of them
    if rqst_branches == 'all':
        ref_branches = reference_data.dtype.names
        smp_branches = sample.data.columns.values
        miss = [b for b in ref_branches if b not in smp_branches]
        # TODO remove this index checker
        miss = [b for b in miss if "_index" not in b]
        assert not miss,\
            "Requested all branches, but did not find {}".format("\n".join(miss))

def test_sample_columns(flat_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data, _ = flat_hits_and_ref
    # Ensure column names are subset of reference names
    check_columns(sample, reference_data)

def test_sample_data(flat_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data, _ = flat_hits_and_ref
    # Ensure all the data is the same
    check_data(sample, reference_data)

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
    Parameterize the flat hit parameters with selections
    """
    file, geom, branches = cstrct_hits_params
    selection = request.param
    selection = " && ".join(NAMES[geom][1]+sel for sel in selection)
    return file, geom, branches, selection

@pytest.fixture()
def flat_hits_sel(cstrct_hits_params_sel):
    """
    Construct the base flat hits object with some selections
    """
    # Unpack the parameters
    file, geom, branches, selection = cstrct_hits_params_sel
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    if branches == "all":
        branches = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample = hits.FlatHits(root_file,
                           tree=tree,
                           prefix=prefix,
                           selection=selection,
                           branches=branches)
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

# SAMPLE SUBSET IMPORT #########################################################

@pytest.fixture(params=[
    # Parameterize the array construction
    # first_event, n_events
    (0, None),
    (10, None),
    (10, 5)
    ])
def cstrct_hits_params_subset(request, cstrct_hits_params):
    """
    Parameterize the flat hit parameters with empty branches
    """
    file, geom, branch = cstrct_hits_params
    first_event, n_events = request.param
    return file, geom, branch, first_event, n_events

@pytest.fixture()
def flat_hits_subset(cstrct_hits_params_subset):
    """
    Construct the base flat hits object with some selections
    """
    # Unpack the parameters
    file, geom, branch, first_event, n_events = cstrct_hits_params_subset
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    if branch == "all":
        branch = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample_all = hits.FlatHits(root_file,
                               tree=tree,
                               prefix=prefix,
                               branches=branch)
    sample_sub = hits.FlatHits(root_file,
                               tree=tree,
                               prefix=prefix,
                               first_event=first_event,
                               n_events=n_events,
                               branches=branch)
    return sample_all, sample_sub, first_event, n_events

def test_flat_hits_subset(flat_hits_subset):
    """
    Test that importing empty branches works fine
    """
    # Unpack the values
    sample_all, sample_sub, f_event, n_events = flat_hits_subset
    # Get information about the whole sample
    all_event_keys, all_n_hits = \
        np.unique(sample_all.get_events()[sample_all.evt_number],
                  return_counts=True)
    all_n_events = all_event_keys.shape[0]
    # Get the expected number of events and index of last event
    if n_events is None:
        n_events = all_n_events - f_event
    l_event = f_event + n_events
    # Get information about the subsample
    sub_event_keys, sub_n_hits = \
        np.unique(sample_sub.get_events()[sample_sub.evt_number],
                  return_counts=True)
    sub_n_events = sub_event_keys.shape[0]
    # Ensure the right number of events are returned
    assert sub_n_events == n_events,\
        "Expected {} events, found {}".format(n_events, sub_n_events)
    # Ensure that the correct first event was returned
    assert sub_event_keys[0] == all_event_keys[f_event],\
        "Expected first event key {} found {}".format(all_event_keys[f_event],
                                                      sub_event_keys[0])
    # Ensure the correct number of hits are returned
    sub_n_hits = np.sum(sub_n_hits)
    expect_n_hits = np.sum(all_n_hits[f_event:l_event])
    assert sub_n_hits == expect_n_hits,\
        "Expected {} hits, found {}".format(expect_n_hits, sub_n_hits)
    # Ensure the event data looks the same
    data_all = sample_all.get_events(events=np.arange(f_event, l_event))
    data_sub = sample_sub.get_events()
    for col in data_sub.columns.values:
        if ("event_index" not in col) and ("hits_index" not in col):
            assert_allclose(data_all[col], data_sub[col], err_msg=col)

# EMPTY BRANCH IMPORT ##########################################################

@pytest.fixture(params=[
    # Parameterize the array construction
    # empty branch name
    ["branch_a"],
    ["branch_b"],
    ["branch_a", "branch_b"]
    ])
def cstrct_hits_params_empty(request, cstrct_hits_params):
    """
    Parameterize the flat hit parameters with empty branches
    """
    file, geom, branch = cstrct_hits_params
    empty = request.param
    empty = [NAMES[geom][1]+sel for sel in empty]
    return file, geom, branch, empty

@pytest.fixture()
def flat_hits_empty(cstrct_hits_params_empty):
    """
    Construct the base flat hits object with some selections
    """
    # Unpack the parameters
    file, geom, branch, empty = cstrct_hits_params_empty
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    if branch == "all":
        branch = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample = hits.FlatHits(root_file,
                           tree=tree,
                           prefix=prefix,
                           empty_branches=empty,
                           branches=branch)
    return sample, empty

def test_flat_hits_empty(flat_hits_empty):
    """
    Test that importing empty branches works fine
    """
    sample, empty = flat_hits_empty
    for empty_branch in empty:
        assert_allclose(sample.data[empty_branch],
                        np.zeros_like(sample.data[empty_branch]),
                        err_msg=empty_branch)

# TEST GET EVENT FUCNTIONS #####################################################

EVENT_LIST = [None, 2, 3, np.arange(5), [1, 3, 7, 10, 11]]

@pytest.fixture(params=list(enumerate(EVENT_LIST)))
def event_params(request):
    """
    Parameters for event functions
    """
    return request.param

@pytest.fixture()
def events_and_ref_data(flat_hits, event_params):
    """
    Return a subsample of events and their reference data
    """
    # Get the event data from the file
    sample, file, geom, _ = flat_hits
    file_index, events = event_params
    # Get the reference file
    ref_file = file + "_" + geom + "_eventdata_"+str(file_index)+".npz"
    # Generate the reference data if needed
    if GENERATE_REFERENCE:
        n_branches = len(sample.data.columns.values)
        generate_reference(ref_file,
                           sample.get_events(events),
                           n_branches,
                           N_BRANCHES[geom])
    ref_data = np.load(ref_file)["array"]
    return sample, ref_data, events

def test_get_event(events_and_ref_data):
    """
    Test if getting specific events works as it is supposed to
    """
    # Unpack the data
    sample, ref_data, events = events_and_ref_data
    event_data = sample.get_events(events)
    for branch in sample.data.columns.values:
        assert_allclose(ref_data[branch], event_data[branch])

def test_get_signal_hits(events_and_ref_data):
    """
    Test if getting specific events works as it is supposed to
    """
    # Unpack the data
    sample, ref_data, events = events_and_ref_data
    event_data = sample.get_signal_hits(events)
    test_ref = ref_data[ref_data[sample.hit_type_name] == True]
    for branch in sample.data.columns.values:
        assert_allclose(test_ref[branch], event_data[branch])

def test_get_background_hits(events_and_ref_data):
    """
    Test if getting specific events works as it is supposed to
    """
    # Unpack the data
    sample, ref_data, events = events_and_ref_data
    event_data = sample.get_background_hits(events)
    test_ref = ref_data[ref_data[sample.hit_type_name] != True]
    for branch in sample.data.columns.values:
        assert_allclose(test_ref[branch], event_data[branch])

@pytest.mark.parametrize("sort_branch, ascending", [
    ([BRANCHES[0], BRANCHES[4], BRANCHES[2]], True),
    ([BRANCHES[3], BRANCHES[1], BRANCHES[5]], True),
    ])
def test_sort_hits(events_and_ref_data, sort_branch, ascending):
    """
    Test if getting specific events works as it is supposed to
    """
    # Unpack the data
    sample, ref_data, events = events_and_ref_data
    # Get the branch names
    evt_branch = sample.prefix+"EventNumber"
    sort_branch = [sample.prefix+srt for srt in sort_branch]
    # Sort the data
    test_ref = np.sort(ref_data, order=[evt_branch, *sort_branch])
    sample.sort_hits(sort_branch, ascending=ascending)
    event_data = sample.get_events(events)
    # Check that it worked
    passing = True
    for branch in sample.data.columns.values:
        # TODO remove this
        if ("hits_index" not in branch):
            error_msg = "Branch {} not sorted".format(branch)
            assert_allclose(test_ref[branch],
                            event_data[branch],
                            err_msg=error_msg)

# TEST FILTERS  ################################################################

@pytest.fixture(params=[
    #variable,      values,       greater, less, invert
    (BRANCHES[0],   None,         0,       None, False),
    (BRANCHES[0],   None,         None,    0,    False),
    (BRANCHES[0],   None,         None,    0,    True),
    ("EventNumber", np.arange(5), None,    None, False)
    ])
def filter_params(request):
    return request.param

def test_filtered_hits(flat_hits, filter_params):
    """
    Keep the hits satisfying this criteria
    """
    # Unpack the arguments
    sample, _, geom, _ = flat_hits
    variable, values, greater, less, invert = filter_params
    variable = NAMES[geom][1] + variable
    # Get the relevant hits to keep
    filtered_hits = sample.filter_hits(variable, these_hits=None,
                                       values=values,
                                       greater_than=greater,
                                       less_than=less,
                                       invert=invert)
    check_filter(filtered_hits, variable, values, greater, less, invert)

def test_trim_hits(flat_hits, filter_params):
    """
    Keep the hits satisfying this criteria
    """
    # Unpack the arguments
    sample, _, geom, _ = flat_hits
    variable, values, greater, less, invert = filter_params
    variable = NAMES[geom][1] + variable
    # Get the relevant hits to keep
    sample.trim_hits(variable,
                     values=values,
                     greater_than=greater,
                     less_than=less,
                     invert=invert)
    check_filter(sample.data, variable, values, greater, less, invert)
