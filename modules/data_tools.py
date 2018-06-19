from __future__ import print_function
from pprint import pprint
import sys
import numpy as np
import scipy
from hits import CDCHits, CTHHits, CyDetHits
# Import from modules
sys.path.insert(0, '../modules')

def data_set_additional_branches(samp,
                                 row_name=None,
                                 cell_id=None,
                                 relative_time=None):
    """
    Set the trigger time and cell ID branches
    """
    if row_name:
        samp.data[row_name] = samp.geom.get_layers(samp.data[samp.flat_name])
    if cell_id:
        samp.data[cell_id] = samp.geom.get_indexes(samp.data[samp.flat_name])
    if relative_time:
        samp.data[relative_time] = samp.data[samp.time_name] - \
                                   samp.data[samp.trig_name]

# Dictionary of cuts to make as we import
def data_get_cuts(needed_cuts, signal=False):
    """
    Naming the cuts in a dictionary to use when importing the file

    Allowed cuts:
        Track = Pass Track quality cuy
        Trig = Pass CTH trigger
        500 = 500 ns lower window
        700 = 700 ns lower window
    """
    cuts = dict()
    geoms = ["CDC", "CTH"]
    t_names = ["DetectedTime", "MCPos.fE"]
    if needed_cuts is None:
        cuts = {"CDC":None,
                "CTH":None}
        return cuts
    # This is getting more complicated than its worth...
    for key, time_name in zip(geoms, t_names):
        prefix = key +"Hit.f"
        time_name = prefix + time_name
        cuts[key] = dict()
        # We already know about signal and triggering
        if signal:
            for key_cut in ["Track", "Trig"]:
                cuts[key][key_cut] = "{}Good{} == 1".format(prefix, key_cut)
        # Here are some timing cuts
        for key_time in ["500", "700"]:
            cuts[key][key_time] = "{} < 1620 && ".format(time_name)+\
            "{} > {}".format(time_name, key_time)
    for geo in geoms:
        cuts[geo] = " && ".join([cuts[geo].get(cut, "1 == 1")
                                 for cut in needed_cuts])
    print("Using cuts")
    pprint(cuts, indent=2)
    return cuts

def data_import_file(file_name, signal=True, use_cuts=None,
                     branches=None, empty_branches=None):
    """
    Import a signal file, with both CTH and CDC hits

    :param file_name: name of the required file
    :use_cuts: name of cuts in cuts dictionary to use
    :branches: branches to import from each tree

    Allowed cuts:
        Track = Pass Track quality cuy
        Trig = Pass CTH trigger
        500 = 500 ns lower window
        700 = 700 ns lower window
    """
    # Get the cuts we asked for
    some_cuts = data_get_cuts(use_cuts, signal=signal)
    if not isinstance(branches, dict) and branches is not None:
        branches = dict()
    # Import the files with the cuts
    print("Getting branches")
    pprint(branches)
    cdc_sample = CDCHits(file_name,
                         tree="CDCHitTree",
                         selection=some_cuts["CDC"],
                         branches=branches.get("CDC", None),
                         empty_branches=empty_branches)
    print("CDC sample, n_events = ", cdc_sample.n_events)
    cth_sample = CTHHits(file_name,
                         tree="CTHHitTree",
                         selection=some_cuts["CTH"],
                         branches=branches.get("CTH", None))
    print("CTH sample, n_events = ", cth_sample.n_events)
    hit_samp = CyDetHits(cdc_sample, cth_sample)
    # Set the trigger time
    if signal:
        hit_samp.cth.set_trigger_time()
    # Remove the smear branch
    return hit_samp

def data_import_sample_no_bkg_cth(this_signal, this_background,
                                  these_cuts=None, branches=None,
                                  empty_branches=None):
    """
    Import both files and keep the number of events in the background sample
    NOTE: we assume the signal sample is larger

    Allowed cuts:
        Track = Pass Track quality cuy
        Trig = Pass CTH trigger
        500 = 500 ns lower window
        700 = 700 ns lower window
    """
    # Import the background CDC hits
    # TODO get the background CTH hits as well!
    if not isinstance(branches, dict) and branches is not None:
        branches = dict()
    # Import the files with the cuts
    print("Getting branches")
    pprint(branches)
    back_cdc_sample = CDCHits(this_background,
                              tree="CDCHitTree",
                              selection=data_get_cuts(these_cuts, signal=False)["CDC"],
                              branches=branches.get("CDC",None),
                              empty_branches=empty_branches)
    sig_hits = data_import_file(this_signal, signal=True,
                                use_cuts=these_cuts, branches=branches,
                                empty_branches=empty_branches)
    # Trim the uncommon events
    sig_hits.keep_common_events()
    # Keep a random number of events
    more_events = max(sig_hits.n_events, back_cdc_sample.n_events)
    less_events = min(sig_hits.n_events, back_cdc_sample.n_events)
    events_to_keep = np.random.permutation(np.arange(0, more_events))[:less_events]
    # Trim the larger set
    if more_events == sig_hits.n_events:
        print("Trimming Signal Events")
        event_numbers = np.unique(sig_hits.cdc.get_events()[sig_hits.cdc.key_name])[events_to_keep]
        sig_hits.cdc.trim_events(event_numbers)
        sig_hits.cth.trim_events(event_numbers)
    else:
        print("Trimming Background Events")
        event_numbers = np.unique(back_cdc_sample.get_events()[back_cdc_sample.key_name])[events_to_keep]
        back_cdc_sample.trim_events(event_numbers)
    print("CTH Sig Events {} ".format(sig_hits.cth.n_events))
    print("CDC Sig Events {} ".format(sig_hits.cth.n_events))
    print("CDC Back Events {} ".format(back_cdc_sample.n_events))
    sig_hits.cdc.add_hits(back_cdc_sample.data)
    sig_hits.set_trigger_time()
    sig_hits.n_events = sig_hits.cdc.n_events
    return sig_hits

def data_import_sample(this_signal, this_background,
                       these_cuts=None, branches=None,
                       empty_branches=None):
    """
    Import both files and keep the number of events in the background sample
    NOTE: we assume the signal sample is larger

    Allowed cuts:
        Track = Pass Track quality cuy
        Trig = Pass CTH trigger
        500 = 500 ns lower window
        700 = 700 ns lower window
    """
    # Import the files
    back_hits = data_import_file(this_background, signal=False,
                                 use_cuts=these_cuts, branches=branches,
                                 empty_branches=empty_branches)
    # TODO hack fix the background cth and cdc to allow for empty cth events
    back_hits.cth.data[back_hits.cth.event_index_name] = \
            back_hits.cth.data[back_hits.cth.key_name] - \
            np.amin(back_hits.cdc.data[back_hits.cdc.key_name])
    back_hits.n_events = max(back_hits.cdc.n_events, back_hits.cth.n_events)
    # Import the signal file
    sig_hits = data_import_file(this_signal, signal=True,
                                use_cuts=these_cuts, branches=branches,
                                empty_branches=empty_branches)
    # Trim the uncommon events
    sig_hits.keep_common_events()
    # Keep a random number of events
    more_events = max(sig_hits.n_events, back_hits.n_events)
    less_events = min(sig_hits.n_events, back_hits.n_events)
    events_to_keep = np.random.permutation(np.arange(0, more_events))[:less_events]
    # Trim the larger set
    if more_events == sig_hits.n_events:
        print("Trimming Signal Events")
        event_numbers = np.unique(sig_hits.cdc.get_events()[sig_hits.cdc.key_name])[events_to_keep]
        sig_hits.cdc.trim_events(event_numbers)
        sig_hits.cth.trim_events(event_numbers)
    else:
        print("Trimming Background Events")
        event_numbers = np.unique(back_hits.cdc.get_events()[back_hits.cdc.key_name])[events_to_keep]
        back_hits.cdc.trim_events(event_numbers)
        back_hits.cth.trim_events(event_numbers)
    print(("CTH Sig Events {} ".format(sig_hits.cth.n_events)))
    print(("CTH Back Events {} ".format(back_hits.cth.n_events)))
    print(("CDC Sig Events {} ".format(sig_hits.cth.n_events)))
    print(("CDC Back Events {} ".format(back_hits.cth.n_events)))
    sig_hits.cdc.add_hits(back_hits.cdc.data)
    sig_hits.cth.add_hits(back_hits.cth.data)
    sig_hits.set_trigger_time()
    sig_hits.n_events = sig_hits.cdc.n_events
    return sig_hits

def data_remove_coincidence(sample, sort_hits=True):
    # Get the energy deposition summed
    all_events = np.arange(sample.cdc.n_events)
    edep_sparse = scipy.sparse.lil_matrix((sample.cdc.n_events,
                                           sample.cdc.geom.n_points))
    sig_hit_sparse = scipy.sparse.lil_matrix((sample.cdc.n_events,
                                              sample.cdc.geom.n_points))
    for evt in all_events:
        # Get the wire_ids of the hit data
        wire_ids = sample.cdc.get_hit_vols(evt, unique=False)
        # Get the summed energy deposition
        edep = np.zeros((sample.cdc.geom.n_points))
        edep_meas = sample.cdc.get_events(evt)[sample.cdc.edep_name]
        np.add.at(edep, wire_ids, edep_meas)
        # Assign this to the sparse array
        edep_sparse[evt, :] = edep
        # Check if there is a signal on this hit
        sig_hit = np.zeros((sample.cdc.geom.n_points))
        sig_hit_meas = sample.cdc.get_events(evt)[sample.cdc.hit_type_name] == \
                      sample.cdc.signal_coding
        np.logical_or.at(sig_hit, wire_ids, sig_hit_meas)
        # Assign this to the sparse array
        sig_hit_sparse[evt, :] = sig_hit
    # Sort by hit type name to keep signal hits preferentialy
    if sort_hits:
        sample.cdc.sort_hits(sample.cdc.time_name, reset_index=True)
    hit_indexes = sample.cdc.get_measurement(sample.cdc.hits_index_name,
                                             all_events)
    # Remove the hits that are not needed
    sample.cdc.trim_hits(sample.cdc.hits_index_name, values=hit_indexes)
    all_events = np.arange(sample.cdc.n_events)
    # Get the wire_ids and event_ids of the hit data
    wire_ids = sample.cdc.get_hit_vols(all_events, unique=False)
    # Map the evnt_ids to the minimal continous set
    evnt_ids = np.repeat(np.arange(all_events.size),
                         sample.cdc.event_to_n_hits[all_events])
    # Force the new edep and is_sig values onto the sample
    hit_indexes = \
        sample.cdc.get_measurement(sample.cdc.hits_index_name,
                                   all_events).astype(int)
    sample.cdc.data[sample.cdc.edep_name][hit_indexes] = \
            edep_sparse[evnt_ids, wire_ids].toarray()
    sample.cdc.data[sample.cdc.hit_type_name][hit_indexes] = \
            sig_hit_sparse[evnt_ids, wire_ids].toarray()

def data_get_measurment_and_neighbours(hit_sample, measurement, events=None, digitize=False, bins=None,
                                       **kwargs):
    """
    Get the measurement on the wire and its neighbours in a classification-friendly way

    :return: a list of three numpy arrays of measurement 1) on wire, 2) to left, 3) to right
    """
    if digitize:
        return [np.digitize(hit_sample.get_measurement(measurement,
                                       events,
                                       shift=i,
                                       only_hits=True,
                                       flatten=True, **kwargs), bins=bins)
                for i in [0,-1,1]]
    else:
        return [hit_sample.get_measurement(measurement,
                                           events,
                                           shift=i,
                                           only_hits=True,
                                           flatten=True, **kwargs)
                    for i in [0,-1,1]]

def data_get_occupancy(cdc_sample):
    """
    Returns sig_occ, back_occ, total_occ
    """
    sig_occ, back_occ, occ = list(), list(), list()
    for event in range(cdc_sample.n_events):
        sig_occ += [len(np.unique(cdc_sample.get_signal_hits(event)[cdc_sample.flat_name]))]
        back_occ += [len(np.unique(cdc_sample.get_background_hits(event)[cdc_sample.flat_name]))]
        occ += [len(np.unique(cdc_sample.get_events(event)[cdc_sample.flat_name]))]

    # print some infor
    avg_n_hits = np.average(cdc_sample.event_to_n_hits)
    avg_occ = np.average(occ)
    print(("Sig Occ: {} , Back Occ : {}".format(np.average(sig_occ), np.average(back_occ))))
    print(("All Occ: {}, {}".format(avg_occ, avg_occ/4482.)))
    print(("NumHits: {}".format(avg_n_hits)))
    print(("MinChansMultiHit: {}".format((avg_n_hits - avg_occ)/float(avg_occ))))

    return sig_occ, back_occ, occ

def data_convert_to_local_cdc(sample):
    # Store the expected names of the positions
    pos = dict()
    for name, samp in zip(["CDC", "CTH"], [sample.cdc, sample.cth]):
        pos[name] = dict()
        for key in ["X", "Y", "Z"]:
            pos[name][key] = samp.prefix + "MCPos.fP.f" + key
        # Reset the values
        samp.data[pos[name]["Z"]] = - (samp.data[pos[name]["Z"]]/10. - 765)
        samp.data[pos[name]["Y"]] = samp.data[pos[name]["Y"]]/10.
        samp.data[pos[name]["X"]] = samp.data[pos[name]["X"]]/10. - 641
    return pos

