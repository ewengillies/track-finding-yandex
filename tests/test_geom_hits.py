"""
Tests for importing data from root file
"""
from __future__ import print_function
#import sys
from root_numpy import list_branches
import pytest
import numpy as np
from numpy.testing import assert_allclose
from test_flat_hits import check_columns, check_data,\
                           filter_branches, generate_reference
from test_flat_hits import FILES, NAMES, BRANCHES, GENERATE_REFERENCE
import hits

# Pylint settings
# pylint: disable=redefined-outer-name

# Number of branches expected for reference samples
N_BRANCHES = {}
N_BRANCHES["CDC"] = 36
N_BRANCHES["CTH"] = 35

# CONSTRUCTOR FIXTURES #########################################################

@pytest.fixture(params=[
    # Parameterize the array construction
    # path, geom, branches,
    (FILES[0], 'all'),
    (FILES[0], BRANCHES)])
def cstrct_geom_hits_params(request):
    """
    Parameterize the geom hit parameters
    """
    return request.param

@pytest.fixture()
def cstrct_cdc_hits_params(cstrct_geom_hits_params):
    """
    Parameterize the cdc hit parameters
    """
    # Unpack file and branches
    file, branches = cstrct_geom_hits_params
    geom = "CDC"
    if branches != 'all':
        branches = [NAMES[geom][1]+branch for branch in BRANCHES]
    return file, geom, branches

@pytest.fixture()
def cstrct_cth_hits_params(cstrct_geom_hits_params):
    """
    Parameterize the cth hit parameters
    """
    # Unpack file and branches
    file, branches = cstrct_geom_hits_params
    geom = "CTH"
    if branches != 'all':
        branches = [NAMES[geom][1]+branch for branch in BRANCHES]
    return file, geom, branches

# TEST CDC HITS ################################################################

@pytest.fixture()
def cdc_hits(cstrct_cdc_hits_params):
    """
    Construct the base cdc hits object
    """
    # Unpack the parameters
    file, geom, rqst_branches = cstrct_cdc_hits_params
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    branches = rqst_branches
    if branches == "all":
        branches = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample = hits.CDCHits(root_file,
                          tree=tree,
                          prefix=prefix,
                          branches=branches)
    # Randomly assign every 5th hit as a signal hit
    sample.data.loc[::5, sample.hit_type_name] = bool(True)
    return sample, file, geom, rqst_branches

@pytest.fixture()
def cdc_hits_and_ref(cdc_hits):
    """
    Package the hits and the reference data together
    """
    # Unpack the parameters
    sample, file, geom, rqst_branches = cdc_hits
    # Check that it is the same as the first time we loaded in this data
    reference_file = file+"_"+geom+"_hits.npz"
    # Generate the reference here if needed
    if GENERATE_REFERENCE:
        generate_reference(reference_file,
                           sample.data,
                           len(sample.data.columns.values),
                           N_BRANCHES[geom])
    reference_data = np.load(reference_file)["array"]
    # Return the information
    return sample, reference_data, rqst_branches

def test_all_cdc_branches_present(cdc_hits_and_ref):
    """
    Ensure we did not drop any branches unintentionally
    """
    # Unpack the values
    sample, reference_data, rqst_branches = cdc_hits_and_ref
    # Ensure we have the right number of branches if we requested all of them
    if rqst_branches == 'all':
        ref_branches = reference_data.dtype.names
        smp_branches = sample.data.columns.values
        miss = [b for b in ref_branches if b not in smp_branches]
        assert not miss,\
            "Requested all branches, but did not find {}".format("\n".join(miss))

def test_cdc_sample_columns(cdc_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data, _ = cdc_hits_and_ref
    # Ensure column names are subset of reference names
    check_columns(sample, reference_data)

def test_cdc_sample_data(cdc_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data, _ = cdc_hits_and_ref
    # Ensure all the data is the same
    check_data(sample, reference_data)

@pytest.fixture(params=[
    # var, events, shift, default, only_hits, flatten, index
    (NAMES["CDC"][1]+BRANCHES[0], None,    None, 0, True,  False, 0), #default
    (NAMES["CDC"][1]+BRANCHES[0], 1,       None, 0, True,  False, 1), #one evt
    (NAMES["CDC"][1]+BRANCHES[0], [1,2,5], None, 0, True,  False, 2), #3 evt
    (NAMES["CDC"][1]+BRANCHES[0], [1,2,5], None, 0, False, False, 3), #empty
    (NAMES["CDC"][1]+BRANCHES[0], [1,2,5], 1,    0, False, False, 4), #pos shft
    (NAMES["CDC"][1]+BRANCHES[0], [1,2,5], -1,   0, False, False, 5), #neg shft
    (NAMES["CDC"][1]+BRANCHES[0], [1,2,5], None, 9, False, False, 6), #diff dft
    (NAMES["CDC"][1]+BRANCHES[0], [1,2,5], None, 9, False, True,  7)])#flat
def cdc_meas_params(request):
    """
    Fixture for checking get measurement functionality
    """
    return request.param

def test_remove_coincidence(cdc_hits):
    """
    Check if we correctly removed the coincidence
    """
    # Unpack the parameters
    sample, file, geom, _ = cdc_hits
    # Get the reference file
    ref_file = file+"_"+geom+"_noconicidence.npz"
    # Get the sample data
    sample.remove_coincidence(sort_hits=True)
    sample_data = sample.get_events()
    # Generate the reference data if needed
    if GENERATE_REFERENCE:
        generate_reference(ref_file,
                           sample_data,
                           len(sample.data.columns.values),
                           N_BRANCHES[geom])
    # Load the reference data
    ref_data = np.load(ref_file)["array"]
    # Check its all the same
    for col in sample.data.columns.values:
        assert_allclose(sample_data[col], ref_data[col], err_msg=col)
        print("PASSED : ", col)

def test_get_measurement(cdc_hits, cdc_meas_params):
    """
    Test the crucial get_measurment function for CDC hits
    """
    # Unpack the parameters
    sample, file, geom, _ = cdc_hits
    var, events, shift, default, only_hits, flatten, index = cdc_meas_params
    # Get the reference file
    ref_file = file + "_" + geom + "_getmeasdata_"+str(index)+".npz"
    # Get the sample data
    sample_data = sample.get_measurement(var,
                                         events=events,
                                         shift=shift,
                                         default=default,
                                         only_hits=only_hits,
                                         flatten=flatten,
                                         use_sparse=False)
    # Generate the reference data if needed
    if GENERATE_REFERENCE:
        generate_reference(ref_file,
                           sample_data,
                           len(sample.data.columns.values),
                           N_BRANCHES[geom])
    # Load the reference data
    ref_data = np.load(ref_file)["array"]
    # Check its all the same
    assert_allclose(sample_data, ref_data, err_msg=var)

# TEST CTH HITS ################################################################

@pytest.fixture()
def cth_hits(cstrct_cth_hits_params):
    """
    Construct the base cth hits object
    """
    # Unpack the parameters
    file, geom, rqst_branches = cstrct_cth_hits_params
    tree, prefix = NAMES[geom]
    root_file = file + ".root"
    # Load all the branches
    branches = rqst_branches
    if branches == "all":
        branches = filter_branches(list_branches(root_file, treename=tree))
    # Load the file
    sample = hits.CTHHits(root_file,
                          tree=tree,
                          prefix=prefix,
                          branches=branches)
    # Randomly assign every 5th hit as a signal hit
    sample.data.loc[::5, sample.hit_type_name] = bool(True)
    return sample, file, geom, rqst_branches

@pytest.fixture()
def cth_hits_and_ref(cth_hits):
    """
    Package the hits and the reference data together
    """
    # Unpack the parameters
    sample, file, geom, rqst_branches = cth_hits
    # Check that it is the same as the first time we loaded in this data
    reference_file = file+"_"+geom+"_hits.npz"
    # Generate the reference here if needed
    if GENERATE_REFERENCE:
        generate_reference(reference_file,
                           sample.data,
                           len(sample.data.columns.values),
                           N_BRANCHES[geom])
    reference_data = np.load(reference_file)["array"]
    # Return the information
    return sample, reference_data, rqst_branches

def test_all_cth_branches_present(cth_hits_and_ref):
    """
    Ensure we did not drop any branches unintentionally
    """
    # Unpack the values
    sample, reference_data, rqst_branches = cth_hits_and_ref
    # Ensure we have the right number of branches if we requested all of them
    if rqst_branches == 'all':
        ref_branches = reference_data.dtype.names
        smp_branches = sample.data.columns.values
        miss = [b for b in ref_branches if b not in smp_branches]
        assert not miss,\
            "Requested all branches, but did not find {}".format("\n".join(miss))

def test_cth_sample_columns(cth_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data, _ = cth_hits_and_ref
    # Ensure column names are subset of reference names
    check_columns(sample, reference_data)

def test_cth_sample_data(cth_hits_and_ref):
    """
    Ensure the data columns are the same
    """
    # Unpack the information
    sample, reference_data, _ = cth_hits_and_ref
    # Ensure all the data is the same
    check_data(sample, reference_data)

@pytest.mark.parametrize("hodoscope", ["both", "down", "up"])
def test_cth_get_events(cth_hits, hodoscope):
    """
    Check the cth can distinguish the up and down hodoscopes well
    """
    # Unpack the information
    sample, file, geom, _ = cth_hits
    # Check that it is the same as the first time we loaded in this data
    reference_file = file+"_"+geom+"_getevents_"+hodoscope+".npz"
    # Generate the reference here if needed
    sample_data = sample.get_events(list(range(3)), hodoscope=hodoscope)
    if GENERATE_REFERENCE:
        generate_reference(reference_file,
                           sample_data,
                           len(sample.data.columns.values),
                           N_BRANCHES[geom])
    ref_data = np.load(reference_file)["array"]
    for col in sample.data.columns.values:
        assert_allclose(sample_data[col], ref_data[col], err_msg=col)
