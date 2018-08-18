"""
Tests for importing data from root file
"""
from __future__ import print_function
#import sys
from math import floor
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
    # Parametrize the array construction
    # path, geom, branches,
    (FILES[0], 'all'),
    (FILES[0], BRANCHES)])
def cstrct_geom_hits_params(request):
    """
    Parametrize the geom hit parameters
    """
    return request.param

@pytest.fixture()
def cstrct_cdc_hits_params(cstrct_geom_hits_params):
    """
    Parametrize the cdc hit parameters
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
    Parametrize the cth hit parameters
    """
    # Unpack file and branches
    file, branches = cstrct_geom_hits_params
    geom = "CTH"
    if branches != 'all':
        branches = [NAMES[geom][1]+branch for branch in BRANCHES]
    return file, geom, branches

# TRIVIAL TESTING ##############################################################

def test_generate_reference():
    """
    Make sure GENERATE_REFERENCE is not left in the ON position without noticing
    """
    assert not GENERATE_REFERENCE,\
        "Generating the reference, most testing is tautologically correct"

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
                           len(sample.all_branches),
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
        smp_branches = sample.all_branches
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
    sample.remove_coincidence()
    sample_data = sample.get_events()
    # Generate the reference data if needed
    if GENERATE_REFERENCE:
        generate_reference(ref_file,
                           sample_data,
                           len(sample.all_branches),
                           N_BRANCHES[geom])
    # Load the reference data
    ref_data = np.load(ref_file)["array"]
    # Check its all the same
    for col in sample.all_branches:
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
                           len(sample.all_branches),
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
                           len(sample.all_branches),
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
        smp_branches = sample.all_branches
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
                           len(sample.all_branches),
                           N_BRANCHES[geom])
    ref_data = np.load(reference_file)["array"]
    for col in sample.all_branches:
        assert_allclose(sample_data[col], ref_data[col], err_msg=col)

@pytest.mark.parametrize("time_bin", [10, 20, 100])
def test_cth_rebin_time(cth_hits, time_bin):
    """
    Check that rebinning in time works for CTH hits
    """
    # Unpack the information
    sample, file, geom, _ = cth_hits
    # Check that the binning is not respected at first
    grouped_data = sample.data.groupby([sample.event_index,
                                        sample.flat_name])
    # Ensure theres is at least one bin with multiple values in all
    # events/channels/bins
    all_unique = True
    for _, group in grouped_data:
        # Skip groups with one hit
        if not group.values.shape[0] > 1:
            continue
        # Get the time values
        time_vals = group[sample.time_name]
        # Get the bins of each time
        time_floor = (time_vals//time_bin) * time_bin
        all_unique = time_floor.shape == np.unique(time_floor).shape
        if not all_unique:
            break
    # Ensure there was one bin which would be resummed
    assert not all_unique,\
        "Time binning {} would combine no hits, trivial test!".format(time_bin)
    # Rebin in time
    sample.rebin_time(t_bin_ns=time_bin)
    grouped_data = sample.data.groupby([sample.event_index,
                                        sample.flat_name])
    # Ensure theres is at least no bins with multiple values in all
    # events/channels/bins
    all_unique = True
    for _, group in grouped_data:
        # Skip groups with one hit
        if not group.values.shape[0] > 1:
            continue
        # Get the time values
        time_vals = group[sample.time_name]
        # Get the bins of each time
        time_floor = (time_vals//time_bin) * time_bin
        assert time_floor.shape == np.unique(time_floor).shape,\
            "Found a time bin where more than one hit exists in the bin"+\
            "time vals : {}".format(group.values)
    # Check that it is the same as the first time we loaded in this data
    reference_file = file+"_"+geom+"_time_bin"+str(time_bin)+".npz"
    # Compare to the reference sample
    if GENERATE_REFERENCE:
        generate_reference(reference_file,
                           sample.data,
                           len(sample.all_branches),
                           N_BRANCHES[geom])
    reference_data = np.load(reference_file)["array"]
    # Check its all the same
    for col in sample.all_branches:
        assert_allclose(sample.data[col], reference_data[col], err_msg=col)
        print("PASSED : ", col)

# TEST CTH TRIGGER #############################################################

@pytest.fixture(params=[
    # Parametrize the array construction
    # t_del, t_win, n_coincidence, max_offset
    (10, 50, 2, 1),
    (20, 100, 2, 1),
    (100, 1000, 2, 1),
    (100, 1000, 3, 1),
    (10, 50, 2, 2),
    (10, 50, 1, 0)])
def trig_params(request):
    """
    Parametrize the trigger
    """
    return request.param

@pytest.fixture()
def cth_with_trig_hits(cth_hits, trig_params):
    """
    Set the trigger for some tests
    """
    # Unpack the information
    sample, file, geom, _ = cth_hits
    # Unpack the information
    t_del, t_win, n_coin, m_off = trig_params
    # Set the trigger pattern
    sample.set_trigger_pattern(n_coincidence=n_coin, max_offset=m_off)
    # Set the trigger hits
    sample.set_trigger_hits(t_win=t_win, t_del=t_del)
    return sample, file, geom, t_del, t_win

@pytest.mark.parametrize("n_proc", [1, 4])
def test_set_trig(cth_with_trig_hits, n_proc):
    """
    Ensure the correct hits are set as trigger hits
    """
    # Unpack the information
    sample, _, _, t_del, t_win = cth_with_trig_hits
    trig_hits = sample.data[sample.data[sample.trig_name]]
    # Ensure its not an empty set
    assert trig_hits.values.size > 0,\
        "Found no trigger hits in any events!"
    # Ensure they were set correctly
    trig_hit_idx = \
        sample.get_trigger_hits(t_win=t_win, t_del=t_del, n_proc=n_proc)
    assert_allclose(trig_hits.index.get_level_values(sample.hit_index),
                    trig_hit_idx)

def test_num_trig_hits(cth_with_trig_hits):
    """
    Ensure the correct number of hits are found
    """
    # Unpack the information
    sample, _, _, _, _ = cth_with_trig_hits
    trig_hits = sample.data[sample.data[sample.trig_name]]
    # Ensure there are atleast the number of hits as required by the pattern
    evt_of_trig_hits = trig_hits.index.get_level_values(sample.event_index)
    _, hits_per_evt = np.unique(evt_of_trig_hits, return_counts=True)
    n_expected = sample.trig_patterns.shape[1]
    assert np.all(hits_per_evt >= n_expected),\
        "Expected at least {} trigger hits in trigger events".format(n_expected)

def test_trig_hits(cth_with_trig_hits):
    """
    Ensure the correct number of hits are found
    """
    # Unpack the information
    sample, _, _, _, t_win = cth_with_trig_hits
    trig_hits = sample.data[sample.data[sample.trig_name]]
    # Ensure there are atleast the number of hits as required by the pattern
    evt_of_trig_hits = trig_hits.index.get_level_values(sample.event_index)
    trig_evts = np.unique(evt_of_trig_hits)
    # Loop through each event and check the properties of the trigger hits
    for evt in trig_evts:
        evt_trig_hits = trig_hits.loc[evt]
        # Ensure they are all trigger hits
        assert np.all(evt_trig_hits[sample.trig_name].values),\
            "Not looking exclusively at trigger hits, but should be!"
        # Ensure the hits make up a trigger pattern
        vol_ids = evt_trig_hits[sample.flat_name]
        # Match the volume ids to a pattern
        match_pat = np.isin(sample.trig_patterns, vol_ids)
        match_pat = np.sum(match_pat, axis=1) == sample.trig_patterns.shape[1]
        assert np.sum(match_pat) == 1,\
            "Volume IDs do not make up exactly one of the required patterns"
        # Ensure all the volume ids are either in the up or down module
        in_one_module = np.all(np.isin(vol_ids, sample.geom.up_crys)) or \
                        np.all(np.isin(vol_ids, sample.geom.down_crys))
        assert in_one_module, "All trigger crystals are in one station"
        # Ensure they fall within the requested time window
        time_hits = evt_trig_hits[sample.time_name]
        max_t, min_t = np.amax(time_hits), np.amin(time_hits)
        assert floor(max_t - min_t) <= t_win,\
            "Expected hits to fall within {} ns time window\n".format(t_win)+\
            " Max : {}\n Min : {}".format(max_t, min_t)

def test_trig_hits_timing(cth_with_trig_hits):
    """
    Ensure the correct number of hits are found
    """
    # Unpack the information
    sample, _, _, t_del, t_win = cth_with_trig_hits
    # Iterated at least once
    compared_once = False
    # Iterate through, trimming off the current trigger hits to ensure that the
    # earliest ones are always found
    while True:
        # Find the timing of each trigger hit
        trig_hits = sample.data[sample.data[sample.trig_name]][sample.time_name]
        trig_time_before = trig_hits.groupby(sample.event_index).min()
        # Remove all the trigger hits
        sample.trim_hits(sample.trig_name, values=False)
        # Remove all the trigger hits
        # Get the trigger hits again
        sample.set_trigger_hits(t_win=t_win, t_del=t_del)
        # Get the timing again
        trig_hits = sample.data[sample.data[sample.trig_name]][sample.time_name]
        trig_time_after = trig_hits.groupby(sample.event_index).min()
        # Ensure that the same number or less trigger events are found the
        # second time
        n_trig_evts_before = np.unique(trig_time_before.index.values).shape[0]
        n_trig_evts_after = np.unique(trig_time_after.index.values).shape[0]
        if n_trig_evts_after == 0:
            break
        assert n_trig_evts_before >= n_trig_evts_after,\
            "Found more trigger events after removing trigger hits!"
        # Iterate through the trigger events the are found the second time
        for evt in np.unique(trig_time_after.index.values):
            # Ensure that each event finds a later trigger signal than before up
            # to nanosecond precision
            trig_t_before = int(trig_time_before.loc[evt])
            trig_t_after = int(trig_time_after.loc[evt])
            assert trig_t_before <= trig_t_after,\
                "Trigger hits found the second time occur earlier than the "+\
                "first time"
            compared_once = True
    assert compared_once, "No trigger events found after removing earlier "+\
        "trigger hits, making this test meaningless."

def test_trig_hits_ref(cth_with_trig_hits):
    """
    Ensure the correct number of hits are found
    """
    # Unpack the information
    sample, file, geom, t_del, t_win = cth_with_trig_hits
    # Get the shape of the trigger patterns
    n_pats, l_pat = sample.trig_patterns.shape
    # Check that it is the same as the first time we loaded in this data
    reference_file = \
        "{}_{}_tdel{}_twin{}_npats{}_lpat{}_trig_hits.npz".format(file, geom,
                                                                  t_del, t_win,
                                                                  n_pats, l_pat)
    # Generate the reference here if needed
    if GENERATE_REFERENCE:
        generate_reference(reference_file,
                           sample.data,
                           len(sample.all_branches),
                           N_BRANCHES[geom]+1)
    reference_data = np.load(reference_file)["array"]
    # Check its all the same
    for col in sample.all_branches:
        assert_allclose(sample.data[col], reference_data[col], err_msg=col)
        print("PASSED : ", col)
