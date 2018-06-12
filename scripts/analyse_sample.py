#!/usr/bin/env python
"""
Executable for trimming down signal
"""

import os
from argparse import ArgumentParser
import time
import data_tools as dts
import visualizations as viz
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

def plot_one_event(args):
    hit_t, (event_id_bkg, event_id_signal), output_dir, geom = args
    to_plot = np.take([0, 2, 1], hit_t)
    axis, fig = viz.plot_output(to_plot,
                                geom,
                                #fig=fig, axis=axis,
                                figsize=(15, 15))
    textstr = "Event ID Background {}".format(event_id_bkg)+\
              "\nEvent ID Signal {}".format(event_id_signal)
    # Create the text if needed, otherwise just update its text
    axis.text(0.005, 0.995, textstr, transform=axis.transAxes,
              verticalalignment='top',
              horizontalalignment='left',
              fontsize=15)
    # Save the file
    file_name = output_dir +\
        "event_bkg-{}_sig-{}.png".format(event_id_bkg, event_id_signal)
    fig.savefig(file_name, bbox_inches='tight')
    fig.clear()


def main():
    parser = ArgumentParser(prog="./analyse_sample.py",
                            description="Check and plot event data")

    parser.add_argument('sig_file',
                        metavar='Signal_File',
                        type=str,
                        help="The tracking tree signal output to be analysed")
    parser.add_argument('back_file',
                        metavar='Background_File',
                        type=str,
                        help="The tracking tree background output to be"+\
                             " analysed")
    parser.add_argument('output_dir',
                        metavar='Output_Directory',
                        type=str,
                        help="The directory to save all results in.")
    parser.add_argument("-n", "--name",
                        dest="name",
                        default="",
                        type=str,
                        help="Prefix for all output file names")
    # Get arguments
    args = parser.parse_args()

    # Check the output directory exists
    if not os.path.isdir(args.output_dir):
        print("Output directory {} does not exist!".format(args.output_dir))
        return 1
    args.output_dir += "/" + args.name

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

    # Plot some features
    bins_for_plots = 50
    # DETECTED TIME CTH #
    _, fig = viz.plot_feature(
        train.cth.get_signal_hits()[train.cth.time_name],
        train.cth.get_background_hits()[train.cth.time_name],
        xlabel="Detected Time [ns]", ylabel="Normalised Hit Count",
        xlog=False,
        title="Detected Time of Hits in CTH",
        nbins=bins_for_plots)
    fig.set_size_inches(10, 5)
    fig.savefig(args.output_dir+"detected_time_cth.png")
    fig.clear()
    # DETECTED TIME CDC #
    _, fig = viz.plot_feature(
        train.cdc.get_signal_hits()[train.cdc.time_name],
        train.cdc.get_background_hits()[train.cdc.time_name],
        xlabel="Detected Time [ns]", ylabel="Normalised Hit Count",
        xlog=False,
        title="Detected Time of Hits in CTH",
        nbins=bins_for_plots)
    fig.savefig(args.output_dir+"detected_time_cdc.png")
    fig.clear()
    # DRIFT TIME #
    _, fig = viz.plot_feature(train.cdc.get_signal_hits()[drift_name],
                              train.cdc.get_background_hits()[drift_name],
                              xlabel="Drift time [ns]",
                              ylabel="Normalised Hit Count",
                              xlog=False,
                              title="Drift Time of Hits in CDC",
                              nbins=bins_for_plots)
    fig.savefig(args.output_dir+"drift_time_cdc.png")
    fig.clear()
    # TRUTH TIME CDC #
    _, fig = viz.plot_feature(
        train.cdc.get_signal_hits()[train.cdc.time_name] -\
            train.cdc.get_signal_hits()[drift_name],
        train.cdc.get_background_hits()[train.cdc.time_name] -\
            train.cdc.get_background_hits()[drift_name],
        xlabel="Truth time [ns]", ylabel="Normalised Hit Count",
        xlog=False,
        title="Truth Time of Hits in CDC",
        nbins=bins_for_plots)
    fig.savefig(args.output_dir+"truth_time_cdc.png")
    fig.clear()
    # RELATIVE TIME CDC #
    _, fig = viz.plot_feature(train.cdc.get_signal_hits()[rel_time_name],
                              train.cdc.get_background_hits()[rel_time_name],
                              xlabel="Relative Time [ns]",
                              ylabel="Normalised Hit Count",
                              xlog=False,
                              title="Relative Time",
                              nbins=bins_for_plots)
    fig.savefig(args.output_dir+"relative_time_cdc.png")
    fig.clear()
    # CHARGE DEPOSITION CDC #
    _, fig = viz.plot_feature(
        np.log10(train.cdc.get_signal_hits()[train.cdc.edep_name] + 1),
        np.log10(train.cdc.get_background_hits()[train.cdc.edep_name] + 1),
        xlabel="log10(Charge Deposition [e])",
        ylabel="Normalised Hit Count",
        xlog=False,
        title="Charge Depostion",
        nbins=bins_for_plots)
    fig.savefig(args.output_dir+"charge_deposition_cdc.png")
    fig.clear()
    # LAYER ID CDC #
    _, fig = viz.plot_feature(train.cdc.get_signal_hits()[row_name],
                              train.cdc.get_background_hits()[row_name],
                              xlabel="Layer ID", ylabel="Normalised Hit Count",
                              xlog=False,
                              title="Layer Number",
                              nbins=18)
    fig.savefig(args.output_dir+"layer_id_cdc.png")
    fig.clear()
    # OCCUPANCIES
    sig_occ, back_occ, occ = train.cdc.get_occupancy()
    plt.title("Occupancy of Events")
    plt.xlabel("% of Wires Hit")
    plt.ylabel("Number of Events / bin")
    plt.hist(np.array(occ)/4482., bins=50)
    fig.savefig(args.output_dir+"total_occupancy_cdc.png")
    fig.clear()
    viz.plot_occupancies(sig_occ, back_occ, occ,
                         n_vols=4482, x_pos=0.2, y_pos=0.8)
    fig.savefig(args.output_dir+"sig_back_occupancy_cdc.png")
    fig.clear()
    # Plot the events
    tstart = time.time()
    events = range(train.cdc.n_events)
    # Get all the event data
    event_ids = train.cdc.get_measurement(train.cdc.key_name,
                                          events=events,
                                          default=-100,
                                          only_hits=False,
                                          flatten=False).astype(int)
    event_ids = np.array([np.unique(some_ids)[1:] for some_ids in event_ids])
    hit_types = train.cdc.get_measurement("CDCHit.fIsSig",
                                          events=events,
                                          default=-1,
                                          only_hits=False,
                                          flatten=False).astype(int) + 1
    data_time = time.time()
    print("Access Time: {} s".format(data_time - tstart))
    # Multiprocess the images
    pool = multiprocessing.Pool()
    pool.map(plot_one_event, zip(hit_types,
                                 event_ids,
                                 [args.output_dir]*train.cdc.n_events,
                                 [train.cdc.geom]*train.cdc.n_events))
    done_time = time.time()
    print("Drawing Time: {} s".format(done_time - data_time))
    print("Total Time: {} s".format(done_time - tstart))

if __name__ == '__main__':
    main()
