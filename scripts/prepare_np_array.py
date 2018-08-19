import sys
import os
sys.path.insert(0, '../modules')
from hits import CDCHits, CTHHits, CDCHits, FlatHits
from pprint import pprint
from collections import Counter
from root_numpy import list_branches
from tracking import HoughSpace
from scipy import sparse
from tracking import HoughTransformer, HoughShifter
from cylinder import CDC
import scipy
import glob
import sys
from data_tools import *
import numpy as np

def set_additional_branches(sample, row_name=None, cell_id=None, relative_time=None):
    """
    Set the trigger time and cell ID branches
    """
    if row_name:
        sample.data[row_name] = sample.geom.get_layers(sample.data[sample.flat_name])
    if cell_id:
        sample.data[cell_id] = sample.geom.get_indexes(sample.data[sample.flat_name])
    if relative_time:
        sample.data[relative_time] = sample.data[sample.time_name] - sample.data[sample.trig_name]

def relabel_position(ret, prefix, title, samp_geom):
    ret[title] = dict()
    pos_list = list()
    for pos in ["P.fX", "P.fY", "P.fZ", "E"]:
        pos_list += [prefix+pos]
    for pos, idx in zip(["x", "y", "z", "t"],
                        [2, 1, 0, 3]):
        ret[title][pos] =  samp_geom.prefix + pos_list[idx]
    samp_geom.data[ret[title]["x"]] = - (samp_geom.data[ret[title]["x"]]/10. - 765)
    samp_geom.data[ret[title]["y"]] =    samp_geom.data[ret[title]["y"]]/10.
    samp_geom.data[ret[title]["z"]] =    samp_geom.data[ret[title]["z"]]/10. - 641

def relabel_momentum(ret, prefix, title, samp_geom):
    ret[title] = dict()
    pos_list = list()
    for pos in ["X", "Y", "Z"]:
        pos_list += [prefix+pos]
    for pos, idx in zip(["x", "y", "z"],
                        [2, 1, 0]):
        ret[title][pos] =  samp_geom.prefix + pos_list[idx]
    samp_geom.data[ret[title]["x"]] = - (samp_geom.data[ret[title]["x"]])

def get_event_labels(evt, a_samp, mom, pos, plot=False):
    # Get the event
    event = a_samp.cdc.get_events(events=evt)
    first_hit = event[0]
    # Get the momentum measurement
    mom_t = np.sqrt(first_hit[mom["hit"]["cdc"]["x"]]**2 +\
                    first_hit[mom["hit"]["cdc"]["y"]]**2)
    mom_z = first_hit[mom["hit"]["cdc"]["z"]]
    # Get the position measurement
    pos_x = first_hit[pos["hit"]["cdc"]["x"]]
    pos_y = first_hit[pos["hit"]["cdc"]["y"]]
    pos_z = first_hit[pos["hit"]["cdc"]["z"]]
    # The the vertex
    vert_x = first_hit[pos["start"]["cdc"]["x"]]
    vert_y = first_hit[pos["start"]["cdc"]["y"]]
    vert_z = first_hit[pos["start"]["cdc"]["z"]]
    # Number of turns
    max_turn_evt = np.unique(event[turn_id_name])[-1]
    if plot:
        entry = {"x" : pos_x, "y" : pos_y, "z" : pos_z}
        make_2d_plots(a_samp.cdc.get_signal_hits(evt),
                      a_samp.cth.get_events(evt),
                      entry, pos)
    return mom_t, mom_z, pos_x, pos_y, pos_z, vert_x, vert_y, vert_z, max_turn_evt

# The most common are stored in these notebooks

# Define some branches to import
## Existing branches
prefix = "CDCHit.f"
drift_name = prefix + "DriftTime"
track_id_name = prefix + "Track.fTrackID"

## Branches to be filled
row_name = prefix +"Layers"
cell_id_name = prefix + "CellID"
rel_time_name = prefix + "Relative_Time"

empty_branches = [row_name, 
                  cell_id_name,
                  rel_time_name]

turn_id_name = prefix + "TurnID"
all_momentum_names = [ prefix + "Track.f" + st_sp + "Momentum.f" + coor 
                       for st_sp in ["Start", "Stop"] for coor in ["X", "Y", "Z"] ]
all_pos_names = [ prefix + "Track.f" + st_sp + "PosGlobal.f" + coor 
                       for st_sp in ["Start", "Stop"] for coor in ["P.fX", "P.fY", "P.fZ", "E"] ]
hit_pos_names = [ prefix + "MCPos.f" + coor for coor in ["P.fX", "P.fY", "P.fZ", "E"] ]
hit_mom_names = [ prefix + "MCMom.f" + coor for coor in ["X", "Y", "Z"] ]

# For track fitting
truth_branches = [turn_id_name] + hit_mom_names + all_momentum_names +\
                 hit_pos_names +  all_pos_names
these_branches = dict()
these_branches["CDC"] = [drift_name, track_id_name] + truth_branches

# Open all of our files
file_root = "/home/five4three2/development/ICEDUST/track-finding-yandex/data/"
file_root = "~/development/ICEDUST/track-finding-yandex/data/MC4p/"
file_root = "/vols/build/comet/users/elg112/ICEDUST/track-finding-yandex/data/MC4p/"
sig_sample_file = file_root + "MC4p_signal_analy_files_500ns.txt"
sig_samples = []
# Read it into a list
with open(sig_sample_file) as f:
    sig_samples= f.read().splitlines()

# Check which file we are processing
file_index = int(sys.argv[1])
sig_file = sig_samples[file_index]
out_file = sig_file.split("/")[-1].split(".")[0]+".npz"
# Import the file
train = data_import_file(sig_file,
                         use_cuts=["500","Trig","Track"],
                         branches=these_branches,
                         empty_branches=empty_branches)
# Set the trigger time
train.set_trigger_time()
# Set the empty branches
set_additional_branches(train.cdc,
                        relative_time=rel_time_name,
                        row_name=row_name,
                        cell_id=cell_id_name)
# Relabel position
pos = dict()
for a_prefix, index in [("MCPos.f", "hit"),
                        ("Track.fStartPosGlobal.f", "start"),
                        ("Track.fStopPosGlobal.f", "stop")]:
    pos[index] = dict()
    relabel_position(pos[index], a_prefix, "cdc", train.cdc)

# Relabel momentum
mom = dict()
for a_prefix, index in [("MCMom.f", "hit"),
                        ("Track.fStartMomentum.f", "start"),
                        ("Track.fStopMomentum.f", "stop")]:
    mom[index] = dict()
    relabel_momentum(mom[index], a_prefix, "cdc", train.cdc)

# Make the labels
# pt, pz, x, y, z, vx, vy, vz, n_turns
labels = np.zeros((train.cdc.n_events, 9))
train.cdc.sort_hits(pos["hit"]["cdc"]["t"])
for evt in range(train.cdc.n_events):
    labels[evt] = get_event_labels(evt, train, mom, pos)

# Remove the coincidence
data_remove_coincidence(train, sort_hits=False)
# Find the events indexes with no backgounds
bkg_hit_evts = np.unique(train.cdc.get_background_hits()[train.cdc.event_index_name])
no_bkg_hit_evts = np.setdiff1d(range(train.cdc.n_events), bkg_hit_evts)
# Get their oaEvent numbers as a set
no_bkg_hit_keys_names = \
        np.unique(train.cdc.get_measurement(train.cdc.key_name, events=no_bkg_hit_evts))
# Remove these events
labels = labels[no_bkg_hit_evts]
train.keep_events(no_bkg_hit_keys_names)

# Initialize the arrays
n_layers = 18
n_points = 300
n_channels = 2
flat_array = np.zeros((train.n_events, n_layers, n_points, n_channels))
# Fill the array
flat_array[train.cdc.get_events()[train.cdc.event_index_name].astype(int),
           train.cdc.get_events()[row_name].astype(int),
           train.cdc.get_events()[cell_id_name].astype(int),
           :] = \
    np.array([train.cdc.get_events()[train.cdc.edep_name],
              train.cdc.get_events()[rel_time_name]]).T

# Shift by the blank cells
shift = (300-train.cdc.geom.n_by_layer).astype(int)//2
for i in range(18):
    flat_array[:,i,:,:] = np.roll(flat_array[:,i,:,:], shift[i], axis=1)
# Set values for the blank cells
for i in range(17):
    flat_array[:,i,:shift[i],:] = -1., -1.
    flat_array[:,i,-shift[i]:,:] = -1., -1.

# Save the numpy file
np.savez_compressed(out_file, image=flat_array, labels=labels)
print("OUTFILE " + out_file)
