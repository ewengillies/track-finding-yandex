#!/usr/bin/env python
"""
Executable for trimming down signal
"""

import sys
from argparse import ArgumentParser
import data_tools as dts
import visualizations as viz

def main():
    parser = ArgumentParser(prog="./analyse_sample.py",
                            description="Check and plot event data")

    parser.add_argument('sig_file',
                        metavar='Signal File',
                        type=str,
                        help="The tracking tree signal output to be analysed")
    parser.add_argument('back_file',
                        metavar='Background File',
                        type=str,
                        help="The tracking tree background output to be"+\
                             " analysed")
    #parser.add_argument("-l", "--layer",
    #                    dest="min_layer",
    #                    default=4,
    #                    type=int,
    #                    help="Minimum max-layer needed for quality track")
    #parser.add_argument("-n", "--n-hits",
    #                    dest="min_hits",
    #                    default=30,
    #                    type=int,
    #                    help="Minimum number of hits for quality track")
    #parser.add_argument("-m", "--min-time",
    #                    dest="min_t",
    #                    default=500.,
    #                    type=float,
    #                    help="Minimum allowed hit time")
    #parser.add_argument("-M", "--max-time",
    #                    dest="max_t",
    #                    default=1170.,
    #                    type=float,
    #                    help="Maximum allowed hit time")
    #parser.add_argument("-d", "--drift",
    #                    dest="drift",
    #                    default=450.,
    #                    type=float,
    #                    help="Maximum allowed CDC drift time for hit")
    #parser.add_argument("-s", "--smear",
    #                    dest="smear",
    #                    default=0.,
    #                    type=float,
    #                    help="Smearing for signal events")
    #parser.add_argument("-o", "--output",
    #                    dest="output",
    #                    default="checked_signal",
    #                    help="Name of the output file")
    #parser.add_argument("-v", "--verbose",
    #                    dest="verbose",
    #                    default=False,
    #                    action="store_true",
    #                    help="Name of the output file")
    # Get arguments
    args = parser.parse_args()
    # Define some branches to import
    ## Existing branches
    prefix = "CDCHit.f"
    drift_name = prefix + "DriftTime"
    track_id_name = prefix + "Track.fTrackID"


    ## Branches to be filled
    row_name = prefix +"Layers"
    cell_id_name = prefix + "CellID"
    rel_time_name = prefix + "Relative_Time"
    take_hit_name = prefix + "Take_Hit"
    lcl_scr_name = prefix + "Local_Score"
    ngh_scr_name = prefix + "Neigh_Score"
    hgh_scr_name = prefix + "Hough_Score"
    trk_scr_name = prefix + "Track_Score"

    empty_branches = [row_name,
                      cell_id_name,
                      lcl_scr_name,
                      ngh_scr_name,
                      hgh_scr_name,
                      trk_scr_name,
                      rel_time_name,
                      take_hit_name]
    # Get some branches
    these_branches = dict()
    these_branches["CDC"] = [drift_name,
                             track_id_name]
    these_branches["CTH"] = None

    # Get the event indexes we will be using
    train = dts.data_import_sample(args.sig_file,
                                   args.back_file,
                                   these_cuts=["500", "Trig", "Track"],
                                   branches=these_branches,
                                   empty_branches=empty_branches)

    dts.data_set_additional_branches(train.cdc,
                                     row_name=row_name,
                                     cell_id=cell_id_name,
                                     relative_time=rel_time_name)

    sig_occ, back_occ, occ = train.cdc.get_occupancy()

    # Plot some features
    bins_for_plots = 50
    viz.plot_feature(train.cth.get_signal_hits()[train.cth.time_name],
                     train.cth.get_background_hits()[train.cth.time_name],
                     xlabel="Detected Time [ns]", ylabel="Normalised Hit Count",
                     xlog=False,
                     title="Detected Time of Hits",
                     nbins=bins_for_plots)

if __name__ == '__main__':
    main()
