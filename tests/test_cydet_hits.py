"""
Tests for importing data from root file
"""
from __future__ import print_function
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
N_BRANCHES["CDC"] = 38
N_BRANCHES["CTH"] = 36

# CONSTRUCTOR FIXTURES #########################################################

@pytest.fixture(params=[
    # Parameterize the array construction
    # path, cdc_remove_events, cth_remove_events, rqst_event
    FILES[0]
    ])
def cstrct_cydet_hits_params(request):
    """
    Parameterize the geom hit parameters
    """
    return request.param

# TEST CYDET HITS ##############################################################

@pytest.fixture()
def cydet_hits(cstrct_cydet_hits_params):
    """
    Construct the base cdc hits object
    """
    # Unpack the parameters
    file = cstrct_cydet_hits_params
    root_file = file + ".root"
    # Load all the branches
    # Load the file
    cdc_tree, cdc_prefix = NAMES["CDC"]
    cdc_branches = filter_branches(list_branches(root_file, cdc_tree))
    cth_tree, cth_prefix = NAMES["CTH"]
    cth_branches = filter_branches(list_branches(root_file, cth_tree))
    # Build the hits class
    cydet = hits.CyDetHits(root_file,
                           cdc_tree=cdc_tree,
                           cdc_prefix=cdc_prefix,
                           cdc_selection=None,
                           cdc_first_event=0,
                           cdc_n_events=None,
                           cdc_branches=cdc_branches,
                           cth_tree=cth_tree,
                           cth_prefix=cth_prefix,
                           cth_selection=None,
                           cth_branches=cth_branches)
    # Return the hits collection
    return cydet

def test_event_key(cydet_hits):
    """
    Ensure the event key is honest
    """
    # Unpack the objects
    cydet = cydet_hits
    # Check the event key before removing any objects
    for evt, (cdc_has, cth_has) in cydet.event_key.iterrows():
        # Get the event in cdc if it has it
        if cdc_has:
            assert cydet.cdc.get_events(events=evt, map_index=False).shape[0] > 0
        #print(evt, cdc_has, cth_has)
