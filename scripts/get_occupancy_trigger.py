#!/usr/bin/env python
"""
Executable Analyzing the Occupancy and Trigger Rate of Events
"""

import os
from argparse import ArgumentParser
import time
from functools import partial
import multiprocessing as mp
import tabulate
from hits import CyDetHits

def analyse_file(file_name,
                 output_dir,
                 first_event,
                 n_events,
                 min_time,
                 max_time,
                 drift_time,
                 n_proc):
    """
    Analyse the file and output the results
    """
    t_start = time.time()
    # Open the output lists
    headers = ["filename"]
    all_vals = [file_name]
    # Import the hits file
    cydet = CyDetHits(file_name,
                      cdc_first_event=first_event,
                      cdc_n_events=n_events,
                      cdc_min_time=min_time,
                      cdc_max_time=max_time,
                      cdc_drift_time=drift_time,
                      cth_min_time=min_time,
                      cth_max_time=max_time,
                      remove_coincidence=False)
    # Get the number of events
    headers += ["n_events_both"]
    all_vals += [cydet.n_events]
    # Print the occupancy
    cdc_occ = cydet.cdc.get_occupancy()
    occ_headers, occ_vals = cydet.cdc.print_occupancy(cdc_occ)
    headers += occ_headers
    all_vals += occ_vals
    # Get the number of triggers
    cydet.cth.set_trigger_hits(n_proc=n_proc)
    n_trig = sum(cydet.cth.get_trigger_events(as_bool_series=True))
    headers += ["n_triggers"]
    all_vals += [n_trig]
    # Add in the number of cth hits
    n_cth_hits = cydet.cth.n_hits
    headers += ["n_cth_hits"]
    all_vals += [n_cth_hits]
    # Add in the number of cth events
    n_cth_events = cydet.cth.n_events
    headers += ["n_cth_events"]
    all_vals += [n_cth_events]
    t_trig = time.time()
    print(tabulate.tabulate(zip(headers, all_vals)))
    print("Total Time: {} s".format(t_trig - t_start))
    # Open the output file
    with open(output_dir+"trig_occupy_stats.txt", "w") as output:
        print(",".join(headers), file=output)
        print(",".join([str(v) for v in all_vals]), file=output)

def main():
    parser = ArgumentParser(prog="./get_occupancy_trigger.py",
                            description="Get the occupancy and trigger rates"+\
                                        " of the sample")
    parser.add_argument('output_dir',
                        metavar='Output Directory',
                        type=str,
                        help="The directory to save all results in.")
    parser.add_argument('files',
                        metavar='Input Files',
                        type=str,
                        nargs='+',
                        help="The tracking tree signal output to be analysed")
    parser.add_argument("-e", "--events",
                        dest="n_events",
                        default=None,
                        type=int,
                        help="Number of events to load")
    parser.add_argument("-f", "--first",
                        dest="first_event",
                        default=0,
                        type=int,
                        help="First event to load")
    parser.add_argument("-P", "--n-parallel",
                        dest="n_parallel",
                        default=1,
                        type=int,
                        help="Number of samples to open in parallel")
    parser.add_argument("-p", "--n-proc",
                        dest="n_proc",
                        default=1,
                        type=int,
                        help="Number of threads to use for the trigger")
    parser.add_argument("-m", "--min-time",
                        dest="min_time",
                        default=None,
                        type=float,
                        help="Minimum time to use for CDC and CTH")
    parser.add_argument("-M", "--max-time",
                        dest="max_time",
                        default=None,
                        type=float,
                        help="Maximum time to use for CTH and CDC (before drift)")
    parser.add_argument("-D", "--drift-time",
                        dest="drift_time",
                        default=450,
                        type=float,
                        help="Allowed drift time in the CDC")
    # Get arguments
    args = parser.parse_args()
    # Check the output directory exists
    if not os.path.isdir(args.output_dir):
        print("Output directory {} does not exist!".format(args.output_dir))
        return 1
    # Append the name to the output file
    prefixes = [file.split("/")[-1].split("_")[1] for file in args.files]
    all_out_dirs = ["{}/{}_".format(args.output_dir, prefix)
                    for prefix in prefixes]
    # Define a partial function with all the arguments fixed
    run_analysis = partial(analyse_file,
                           first_event=args.first_event,
                           n_events=args.n_events,
                           min_time=args.min_time,
                           max_time=args.max_time,
                           drift_time=args.drift_time,
                           n_proc=args.n_proc)
    # If its only one file at a time or one file
    if (args.n_parallel == 1) or (len(args.files) == 1):
        # Iterate through each file one at a time
        for file, outdir in zip(args.files, all_out_dirs):
            run_analysis(file, outdir)
    else:
        to_open = min(args.n_parallel, len(args.files))
        with mp.Pool(processes=to_open) as pool:
            pool.starmap(run_analysis, zip(args.files, all_out_dirs))
    #analyse_file(args.file

if __name__ == '__main__':
    main()
