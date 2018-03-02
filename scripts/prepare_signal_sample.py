#!/usr/bin/env python
"""
Executable for trimming down signal
"""

from collections import OrderedDict
from argparse import ArgumentParser
import numpy as np
from root_numpy import root2array
from rootpy.tree import Tree, TreeModel, TreeChain, BoolCol, FloatCol
from rootpy.io import root_open
from hits import CyDetHits, CTHHits

def make_random_time(signal_file, smear):
    rand_t = OrderedDict()
    all_events = np.union1d(root2array(signal_file, "CTHHitTree",
                                       branches="CTHHit.fEventNumber"),
                            root2array(signal_file, "CDCHitTree",
                                       branches="CDCHit.fEventNumber"))
    for event in all_events:
        rand_t[event] = np.random.uniform(-smear, smear)
    return rand_t

def import_cth_sample(cth_file, rand_t, min_t=500, max_t=1170):
    # Import the hits
    cth_samp = CTHHits(cth_file, tree="CTHHitTree")
    # Smear the CTH time
    cth_samp.data[cth_samp.time_name] += \
            np.vectorize(rand_t.get)(cth_samp.get_events()[cth_samp.key_name])
    # Remove the hits outside the time window
    cth_samp.trim_hits(variable=cth_samp.time_name,
                       greater_than=min_t,
                       less_than=max_t)
    # Get the trigger time
    cth_samp.set_trigger_time()
    return np.unique(cth_samp.data[cth_samp.get_trig_hits()][cth_samp.key_name])

def import_cdc_sample(cdc_file, rand_t, min_t=500, max_t=1620,
                      min_hits=30, min_layer=4):
    # Import the hits
    cdc_samp = CyDetHits(cdc_file, tree="CDCHitTree",
                         selection="CDCHit.fIsSig == 1")
    # Smear the CDC time
    cdc_samp.data[cdc_samp.time_name] += \
            np.vectorize(rand_t.get)(cdc_samp.get_events()[cdc_samp.key_name])
    cdc_samp.trim_hits(variable=cdc_samp.time_name,
                       greater_than=min_t,
                       less_than=max_t)
    # Get the events that pass the cuts
    enough_hits = cdc_samp.min_hits_cut(min_hits)
    enough_layer = cdc_samp.min_layer_cut(min_layer)
    return np.intersect1d(enough_hits, enough_layer)


class ExtraBranches(TreeModel):
    GoodTrack = BoolCol()
    GoodTrig = BoolCol()
    SmearTime = FloatCol()

def copy_in_trigger_signal(in_files_name, out_name, tree_name,
                           prefix, cdc_events, cth_events, rand_t):
    # Convert input lists to sets first
    set_cdc_events = set(cdc_events)
    set_cth_events = set(cth_events)

    # Define the chain of input trees
    in_chain = TreeChain(name=tree_name, files=in_files_name)
    # First create a new file to save the new tree in:
    out_file = root_open(out_name, "r+")
    out_tree = Tree(tree_name, model=ExtraBranches.prefix(prefix))

    # This creates all the same branches in the new tree but
    # their addresses point to the same memory used by the original tree.
    out_tree.create_branches(in_chain._buffer)
    out_tree.update_buffer(in_chain._buffer)

    # Now loop over the original tree(s) and fill the new tree
    for entry in in_chain:
        # Add in the new values
        this_event_number = entry[prefix+"EventNumber"].value
        out_tree.__setattr__(prefix+"GoodTrack",
                             this_event_number in set_cdc_events)
        out_tree.__setattr__(prefix+"GoodTrig",
                             this_event_number in set_cth_events)
        try:
            out_tree.__setattr__(prefix+"SmearTime", rand_t[this_event_number])
        except:
            for key, item in entry.iteritems():
                print key, item
        # Fill, noting that most of the buffer is shared between the chain
        # and the output tree
        out_tree.Fill()
    # Close it up
    out_tree.Write()
    out_file.Close()

def main():
    parser = ArgumentParser(prog="./prepare_signal_sample.py",
                            description="Check events and label "+\
                                        " for trigger signal and "+\
                                        " track quality")
    parser.add_argument('input_file',
                        metavar='Input File',
                        type=str,
                        help="The tracking tree output that must be "+\
                             "checked for signal")
    parser.add_argument("-l", "--layer",
                        dest="min_layer",
                        default=4,
                        help="Minimum max-layer needed for quality track")
    parser.add_argument("-n", "--n-hits",
                        dest="min_hits",
                        default=30,
                        help="Minimum number of hits for quality track")
    parser.add_argument("-m", "--min-time",
                        dest="min_t",
                        default=500.,
                        help="Minimum allowed hit time")
    parser.add_argument("-M", "--max-time",
                        dest="max_t",
                        default=1170.,
                        help="Maximum allowed hit time")
    parser.add_argument("-d", "--drift",
                        dest="drift",
                        default=450.,
                        help="Maximum allowed CDC drift time for hit")
    parser.add_argument("-s", "--smear",
                        dest="smear",
                        default=50.,
                        help="Smearing for signal events")
    parser.add_argument("-o", "--output",
                        dest="output",
                        default="checked_signal",
                        help="Name of the output file")
    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        default=False,
                        action="store_true",
                        help="Name of the output file")
    # Get arguments
    args = parser.parse_args()


    # Create an ordered dictionary of event IDs to random time smears
    random_time = make_random_time(args.input_file, args.smear)
    # Get the event indexes we will be using
    good_cdc_events = import_cdc_sample(args.input_file, random_time,
                                        min_t=args.min_t,
                                        max_t=args.max_t+args.drift,
                                        min_hits=args.min_hits,
                                        min_layer=args.min_layer)
    good_cth_events = import_cth_sample(args.input_file, random_time,
                                        min_t=args.min_t,
                                        max_t=args.max_t)
    copy_in_trigger_signal(args.input_file, args.output,
                           "CDCHitTree", "CDCHit.f",
                           good_cdc_events, good_cth_events,
                           random_time)
    copy_in_trigger_signal(args.input_file, args.output,
                           "CTHHitTree", "CTHHit.f",
                           good_cdc_events, good_cth_events,
                           random_time)

if __name__ == '__main__':
    main()
