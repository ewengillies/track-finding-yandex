"""
Notation used below:
 - wire_id is flat enumerator of all wires
 - layer_id is the index of layer
 - wire_index is the index of wire in the layer
"""
from __future__ import print_function
from math import ceil
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import dask
import tabulate
from dask import dataframe as dd
from cylinder import CDC, CTH
from uproot_selected import import_uproot_selected, check_for_branches

# TODO swith to pandas
    # TODO move data_tools into hits
    # TODO deal with multi-indexing events from evt_number
        # TODO deal with empty CTH events or empty CDC events:
        #      sparse DF with both CTH and CDC hits.
        #      * do messy version for now, just need:
        #         * adding them to work
        #         * returning results one event at a time
# TODO run the analysis once
# TODO change all documentation

def _is_sequence(arg):
    """
    Check if the argument is iterable
    """
    # Check for strings
    if hasattr(arg, "strip"):
        return False
    # Check for numpy scalars
    if np.isscalar(arg):
        return False
    # Check for iterable or indexable items
    if hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"):
        return True
    # Return false otherwise
    return False

def _return_branches_as_list(branches):
    """
    Ensures branches are given as list
    """
    # Deal with requested branches
    if branches is None:
        branches = []
    elif not _is_sequence(branches):
        branches = list(branches)
    return list(branches)

def map_indexes(old_indexes, new_indexes, force_shape=True):
    # TODO documentation, more
    """
    Map the values in the old_indexes to the unique values in the new_indexes.
    This mapping preserves the order in which indexes occur in each set.
    """
    # Get the information needed for the mapping, which is the correspondence
    # mapping between the two sets of indexes, as generated from numpy.unique
    v_old, idx_old, inv_old = np.unique(old_indexes,
                                        return_index=True,
                                        return_inverse=True)
    v_new, idx_new = np.unique(new_indexes,
                               return_index=True)
    # Preserve the order of the indexes as they first appeared
    new_idx_vals = v_new[idx_new.argsort()]
    old_idx_vals = v_old[idx_old.argsort()]
    # Ensure there are enough new indexes to map the old ones
    assert v_new.shape[0] >= v_old.shape[0],\
        "Not enough new indexes to map to the old ones"+\
        "Number of unique new indexes {}".format(v_new.shape[0])+\
        "Number of unique old indexes {}".format(v_old.shape[0])
    # The new index values can be longer than the old, but not vice-versa!
    if force_shape:
        new_idx_vals = new_idx_vals[:v_old.shape[0]]
    # Get the mappings from old keys to new indexes with the order that each
    # occured first preserved
    map_v = dict(zip(old_idx_vals, new_idx_vals))
    # Map the old values to the new values, then use the inverse mapping give
    # from numpy unique for the old indexes
    return np.array([map_v[x] for x in v_old]).astype(np.int64)[inv_old]

class FlatHits():
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self, path, tree, prefix,
                 selection=None,
                 first_event=0,
                 n_events=None,
                 branches=None,
                 hit_type_name="IsSig",
                 evt_number="EventNumber",
                 hit_number="HitNumber"):
        # TODO fill in docs
        """
        """
        # Column naming and signal labelling conventions
        # The prefix comes before all column names
        self.prefix = prefix
        # The event and hit numbers determines the origin of the datum in the
        # input data files
        self.evt_number = self.prefix + evt_number
        self.hit_number = self.prefix + hit_number
        # The hit type name denotes if the hit is signal or not
        self.hit_type_name = self.prefix + hit_type_name
        # Each hit by hit ID and by event index, where event index is contiguous
        # and starts from zero
        self.hit_index = "hit_index"
        self.event_index = "event_index"

        # The selection controls which hits are imported
        self.selection = selection
        # Add the branches as we expect
        self._branches = _return_branches_as_list(branches)
        self._branches += [self.hit_type_name, self.evt_number]

        # Get the number of hits for each event, limiting this lookup table if
        # we only want a subset of events
        self.n_events, self.n_hits, _hit_sel_str = \
            self._count_hits_events(path, tree, first_event, n_events)
        # Add the hit selection into the selection string
        if _hit_sel_str:
            if self.selection is not None:
                self.selection += " && " + _hit_sel_str
            else:
                self.selection = _hit_sel_str
        # Finialize the data into the data structures
        self.data = self._finalize_data(path, tree)
        self._reset_indexes(evt_index=self.data[self.evt_number].values,
                            construction=True)
    @property
    def all_branches(self):
        return self.data.columns.values

    def _import_root_file(self, path, tree, branches, selection=None,
                          single_perc=True, num_entires=None):
        """

        :param path: path to root file
        :param tree: name of tree in root file
        :param branches: required branches
        """
        # TODO docs
        # Ensure branches is a list
        if not _is_sequence(branches):
            branches = [branches]
        # Add the prefix
        branches = [self.prefix + b
                    if self.prefix not in b else b
                    for b in branches]
        branches = sorted(list(set(branches)))
        # Check the branches we want are there
        check_for_branches(path, tree, branches)
        # Allow for custom selections
        if selection is None:
            selection = self.selection
        # Import the branches
        data = import_uproot_selected(path, tree, branches,
                                      selection=selection,
                                      num_entries=num_entires,
                                      single_perc=single_perc)
        # Return the data frame
        return data

    def _count_hits_events(self, path, tree, first_event=0, n_events=None):
        """
        Creates look up tables to map from event index to number of hits from
        the root files
        """
        # Check the branch we need to define the number of hits is there
        _import_branches = [self.evt_number]
        check_for_branches(path, tree, branches=_import_branches)
        # Import the data
        e_data = import_uproot_selected(path, tree,
                                        branches=_import_branches,
                                        selection=self.selection)
        e_data = e_data[self.evt_number]
        # Check the hits are sorted by event
        e_data_sort = np.sort(e_data)
        assert np.array_equal(e_data, np.sort(e_data)),\
            "Event index named {} not sorted".format(self.evt_number)
        # Return the number of hits in each event
        _, evt_n_hits = np.unique(e_data, return_counts=True)
        # Denote which hits will be imported as a selection string so it plays
        # nice with other selection strings
        _hit_sel = []
        # Default case for the last event number of events, which defaults to
        # all remaining events (i.e. index of None)
        last_event = n_events
        if first_event != 0:
            # Incriment the index of the last event so that it equals
            # n_events + first_event if both are defined, or remains None
            # otherwsie
            if last_event is not None:
                last_event += first_event
            # Select the first hit from this event
            sel = "({} >= {})".format(self.evt_number,
                                      e_data[np.sum(evt_n_hits[:first_event])])
            _hit_sel += [sel]
        # Check which the last is
        if last_event is not None:
            # Get the last hit and add it to the selection
            sel = "({} < {})".format(self.evt_number,
                                     e_data[np.sum(evt_n_hits[:last_event])])
            _hit_sel += [sel]
        # Remove the events we don't care about
        evt_n_hits = evt_n_hits[first_event:last_event]
        # Return the number of events, the number of hits, and the hit selection
        # information
        return evt_n_hits.shape[0], np.sum(evt_n_hits), " && ".join(_hit_sel)

    def _check_consistent_index(self, idx_new):
        # TODO documentation
        """
        Checks there is a one-to-one mapping between old and new indexes
        """
        # Get the old index
        old_evt_idx = self.data.index.get_level_values(self.event_index).values
        # Map the old values onto the new values, this should really just
        # recover old_evt_idx if they are consistent!
        new_to_old = map_indexes(idx_new, old_evt_idx, force_shape=False)
        # Check that the old mapping can be recovered from the new one
        np.testing.assert_array_equal(old_evt_idx, new_to_old)

    def set_event_indexes(self, new_indexes):
        # TODO docuumentation
        """
        """
        # Get the old index
        old_evt_idx = self.data.index.get_level_values(self.event_index).values
        # Map the old values onto the new values, this should really just
        # recover old_evt_idx if they are consistent!
        _new_indexes = map_indexes(old_evt_idx, new_indexes)
        self._reset_indexes(evt_index=_new_indexes)

    def _reset_indexes(self, evt_index=None, construction=False, sort_index=True):
        # TODO docuumentation
        """
        """
        if sort_index:
            # Sort by the new index
            self.data.sort_index(level=[self.event_index,
                                        self.hit_index], inplace=True)
        # If no values are passed, use the existing event index
        if evt_index is None:
            evt_index = \
                self.data.index.get_level_values(self.event_index).values
        # Check the old index is consistent with the new one
        elif not construction:
            self._check_consistent_index(evt_index)
        # If so, get the unique values
        _, event_to_n_hits = np.unique(evt_index, return_counts=True)
        # Set the event index
        self.data.loc[:, self.event_index] = evt_index.astype(np.int64)
        # Set the hit index
        hit_idx = np.arange(np.sum(event_to_n_hits))
        self.data.loc[:, self.hit_index] = hit_idx.astype(np.int64)
        # Set the indexes on the data frame
        self.data.set_index([self.event_index, self.hit_index],
                             inplace=True, drop=True)
        # Set the number of hits and events as well
        self.n_hits = sum(event_to_n_hits)
        self.n_events = event_to_n_hits.shape[0]
        # Sort by the new index
        if sort_index:
            self.data.sort_index(level=[self.event_index,
                                        self.hit_index], inplace=True)

    def _finalize_data(self, path, tree):
        """
        Zip up the data into a rec array if this is the highest level class of
        this instance
        """
        # Import the branches we expect
        return self._import_root_file(path, tree, branches=self._branches)

    def _get_mask(self, these_hits, variable, values=None, greater_than=None,
                  less_than=None, invert=False):
        """
        Returns the section of the data where the variable equals
        any of the values
        """
        # Default is all true
        mask = np.ones(len(these_hits))
        if not values is None:
            # Switch to a list if a single value is given
            if not _is_sequence(values):
                values = [values]
            this_mask = these_hits[variable].isin(values)
            mask = np.logical_and(mask, this_mask)
        if not greater_than is None:
            this_mask = these_hits[variable] > greater_than
            mask = np.logical_and(mask, this_mask)
        if not less_than is None:
            this_mask = these_hits[variable] < less_than
            mask = np.logical_and(mask, this_mask)
        if invert:
            mask = np.logical_not(mask)
        return mask

    def get_events(self, events=None, map_index=True):
        """
        Returns the hits from the given event(s).  Default gets all events

        :param unique: Force each event to only be retrieved once
        """
        # Return the data for these events
        if events is None:
            return self.data
        # Return all events by default
        loc_val = events
        if map_index:
            loc_lvl = self.data.index.names.index(self.event_index)
            loc_val = self.data.index.levels[loc_lvl][loc_val]
        # Ensure its still a sequence so a multiindex data frame is returned
        if not _is_sequence(loc_val):
            loc_val = [loc_val]
        return self.data.loc[loc_val]

    def keep_events(self, event_index):
        """
        Keep the events given by the index in data.  Note this uses the index
        values, not the index value order.
        """
        # Get a boolean mask of the hits to keep by event number
        to_keep_mask = self.data.index.isin(event_index, level=self.event_index)
        # Remove the events that arent needed
        self.data = self.data[to_keep_mask]
        # Reset the indexes
        self._reset_indexes()

    def sort_hits(self, variable=None, ascending=True, reset_index=True):
        """
        Sorts the hits by the given variable inside each event.  By default,
        this is done in acending order and the hit index is reset after sorting.
        """
        # Always sort by event index
        var_list = [self.event_index]
        # If variable(s), sort by this variable as well
        if variable is not None:
            # Unpack it if its a sequence
            if _is_sequence(variable):
                var_list = [self.event_index, *variable]
            else:
                var_list = [self.event_index, variable]
        # Sort each event internally
        self.data.sort_values(var_list, ascending=ascending, inplace=True)
        # Reset the hit index
        if reset_index:
            self._reset_indexes(sort_index=False)

    def filter_hits(self, variable, these_hits=None,
                    values=None, greater_than=None,
                    less_than=None, invert=False):
        """
        Returns the section of the data where the variable equals
        any of the values
        """
        if these_hits is None:
            these_hits = self.get_events()
        mask = self._get_mask(these_hits=these_hits,
                              variable=variable,
                              values=values,
                              greater_than=greater_than,
                              less_than=less_than,
                              invert=invert)
        return these_hits[mask]

    def keep_hits_where(self, variable, values=None, greater_than=None,
                        less_than=None, invert=False):
        """
        Keep the hits satisfying this criteria
        """
        # Get the relevant hits to keep
        self.data = self.filter_hits(variable,
                                     these_hits=self.data,
                                     values=values,
                                     greater_than=greater_than,
                                     less_than=less_than,
                                     invert=invert)
        self._reset_indexes()

    def add_events(self, hits_to_add):
        # TODO documentation
        """
        Append events to the end of an existing sample
        """
        # Incriment the indexes of the sample to add so that they are higher
        # than the existing event indexes
        loc_lvl = self.data.index.names.index(self.event_index)
        max_index = np.amax(self.data.index.levels[loc_lvl].values)
        # Set the indexes of the data to add to these new indexes
        new_indexes = np.arange(max_index, max_index + hits_to_add.n_events)
        # Reset the index, adding one so the range is
        # (max_index, max_index + n_events]
        hits_to_add.set_event_indexes(new_indexes + 1)
        # Append the data as is to the old data
        self.data = self.data.append(hits_to_add.data)
        self._reset_indexes()

    def add_hits(self, hits_to_add, fix_indexes=True):
        # TODO documentation
        """
        Add hits of two samples to join events
        """
        # Determine which sample has more events and therefore which indexes to
        # remap, defaulting to the assumption that self.data has more events
        origin, to_add = self, hits_to_add
        # Fix the indexes so that the addition goes smoothly.  This should be
        # (and is) the default
        if fix_indexes:
            if hits_to_add.n_events > self.n_events:
                origin, to_add = self, hits_to_add
            # Remap the indexes of the smaller set to match the indexes of the
            # larger set
            new_evt_idx = origin.data.index.get_level_values(self.event_index)
            # Set the indexes of the data to add to these new indexes
            to_add.set_event_indexes(np.unique(new_evt_idx))
        # Append the data as is to the old data
        self.data = origin.data.append(to_add.data)
        self._reset_indexes()

    # TODO depreciate
    def remove_branch(self, branch_names):
        """
        Remove a branch from the data
        """
        if not _is_sequence(branch_names):
            branch_names = [branch_names]
        all_names = list(self.data.dtype.names)
        for branch in branch_names:
            prefixed_branch = self.prefix + branch
            if branch in all_names:
                all_names.remove(branch)
            elif prefixed_branch in all_names:
                all_names.remove(prefixed_branch)
        self.data = self.data[all_names]

    def get_signal_hits(self, events=None):
        """
        Returns the hits from the same event(s) as the given hit list.
        Default gets hits from all events.
        """
        # Get the events
        these_hits = self.filter_hits(self.hit_type_name,
                                      these_hits=self.get_events(events),
                                      values=True, invert=False)
        return these_hits

    def get_background_hits(self, events=None):
        """
        Returns the hits from the same event(s) as the given hit list
        Default gets hits from all events.
        """
        these_hits = self.filter_hits(self.hit_type_name,
                                      these_hits=self.get_events(events),
                                      values=True, invert=True)
        return these_hits

    def print_branches(self):
        """
        Print the names of the data available once you are done
        """
        # Print status message
        print("Branches available are:")
        print("\n".join(self.all_branches))

class GeomHits(FlatHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    # pylint: disable=unbalanced-tuple-unpacking
    def __init__(self, geom, path, tree, prefix,
                 branches=None,
                 edep_name="Charge",
                 time_name="MCPos.fE",
                 flat_name="Channel",
                 selection=None,
                 min_time=None,
                 max_time=None,
                 **kwargs):
        """
        This generates hit data in a structured array from an input root file
        from a file. It assumes that the hits are associated to some geometrical
        structure, which is organized by row and index.

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        :param branches: branches from root file to import
        :param hit_type_name: name of the branch that determines hit type
        :param n_hit_name: name of branch that gives the number of hits in each
                           event
        """
        # Define the names of the time and energy depostition columns
        branches = _return_branches_as_list(branches)
        self.edep_name = prefix + edep_name
        self.time_name = prefix + time_name
        branches += [self.edep_name, self.time_name]
        # Check if there are any cuts on time
        for time_val, oper in zip([min_time, max_time],
                                  [">=", "<="]):
            if time_val is not None:
                time_cut = "{} {} {}".format(self.time_name, oper, time_val)
                if selection is not None: 
                    selection += " && " + time_cut
                else:
                    selection = time_cut
        # Define the flattened indexes of the geometry
        self.flat_name = prefix + flat_name
        # Get the geometry of the detector
        self.geom = geom
        # Initialize the base class
        FlatHits.__init__(self, path, tree, prefix,
                          selection=selection,
                          branches=branches, 
                          **kwargs)

    def set_layer_info(self, row_name="Layer"):
        """
        Set the trigger time and cell ID branches
        """
        # TODO documentation
        # Add the prefix
        row_name = self.prefix + row_name
        # Map the values from the geometry object
        self.data[row_name] = self.geom.get_layers(self.data[self.flat_name])
        return row_name

    def set_relative_time(self, trigger_times, rel_time_name="RelativeTime"):
        """
        Set the relative time of the hit to the trigger signal
        """
        # TODO documentation, note that trigger_times needs to have an entry for
        # all evt_idxs in self
        # Add the prefix
        rel_time_name = self.prefix + rel_time_name
        # Get the event index of each hit
        evt_idxs = self.data.index.get_level_values(self.event_index)
        # Calculate the relative time column
        rel_time = self.data[self.time_name] - trigger_times.loc[evt_idxs]
        self.data[rel_time_name] = rel_time

    def get_measurement(self, events, name):
        """
        Returns requested measurement by event

        :return: numpy.array of length self.n_hits
        """
        # Select the relevant event from data
        return self.get_events(events)[name]

    def get_hit_vector(self, events=None, only_back=False):
        """
        Get hits in all volumes by type, 1 is signal, 2 in background, nothing
        is 0. Signal takes priority.

        :return: numpy.array of shape [n_events, Geometry.n_points]
        """
        # TODO documentation
        # Check if events is none
        if events is None:
            events = np.arange(self.n_events)
        # Allow for a single event
        if not _is_sequence(events):
            events = [events]
        # Count how many events there are
        n_events = len(events)
        # Get the flat vol IDs for those vols with sig or bkg hits
        vols = self.get_events(events)[[self.flat_name, self.hit_type_name]]
        # Sort to ensure the signal hits come last
        vols.sort_values([self.event_index,
                          self.flat_name,
                          self.hit_type_name],
                          ascending=True,
                          inplace=True)
        # Initialize the return values
        vol_vect = np.zeros((n_events, self.geom.n_points), dtype=np.uint8)
        # Get the event and volume indexes to map the result
        vol_idxs = vols[self.flat_name].values
        evt_idxs = vols.index.get_level_values(self.event_index)
        evt_idxs = map_indexes(evt_idxs, np.arange(n_events))
        # Mark the ones with a hit by the hit type
        vol_vect[evt_idxs, vol_idxs] = np.take([2, 1], vols[self.hit_type_name])
        return vol_vect

    def get_energy_deposits(self, events):
        """
        Returns energy deposit in all wires

        :return: numpy.array of shape [CDC.n_points]
        """
        return self.get_measurement(events, self.edep_name)

    def get_hit_time(self, events):
        """
        Returns the timing of the hit

        :return: numpy.array of shape [CDC.n_points]
        """
        return self.get_measurement(events, self.time_name)

class CDCHits(GeomHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, path,
                 geom=CDC(),
                 tree="CDCHitTree",
                 prefix="CDCHit.f",
                 time_name="DetectedTime",
                 branches=None,
                 flat_name="Channel",
                 max_time=None,
                 drift_time=450.,
                 **kwargs):
        """
        This generates hit data in a structured array from an input root file
        from a file. It assumes the naming convention "CDCHit.f"+ variable for
        all leaves. It overlays its data on the uses the CDC class to define
        its geometry.

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        :param branches: branches from root file to import
        :param hit_type_name: name of the branch that determines hit type
        :param n_hit_name: name of branch that gives the number of hits in each
                           event
        """
        # Add the flat name to the branches
        branches = _return_branches_as_list(branches) + [prefix + flat_name]
        if (drift_time is not None) and (max_time is not None):
            max_time += drift_time
        # Build the geom hits object
        GeomHits.__init__(self, geom, path, tree, prefix,
                          branches=branches,
                          time_name=time_name,
                          flat_name=flat_name,
                          max_time=max_time,
                          **kwargs)
        # Set the layer information
        self.row_name = self.set_layer_info()
        # Sort the hits by time
        self.sort_hits(self.time_name)

    def remove_coincidence(self):
        """
        Removes the coincidence by:
            * Summing the energy deposition
            * Taking a signal hit label if it exists on the channel
            * Taking the rest of the values from the earliest hit on the channel
        """
        # Sort the hits by channel and by time
        self.sort_hits([self.flat_name, self.time_name])
        # Ensure we sum the energy deposition and keep the signal labels
        agg_dict = {branch : 'first' for branch in self.all_branches}
        agg_dict[self.edep_name] = 'sum'
        agg_dict[self.hit_type_name] = 'any'
        # Group by the channels
        group_list = [self.event_index, self.flat_name]
        grp_data = self.data.groupby(group_list, sort=False)
        # Aggregate the data and reset the indexes
        self.data = grp_data.agg(agg_dict)
        self.data.reset_index(level=group_list[1:], drop=True, inplace=True)
        # Sort the hits by time again
        self.sort_hits(self.time_name)

    def get_measurement(self, name,
                        events=None,
                        shift=None,
                        default=0,
                        dtype=float,
                        only_hits=True,
                        flatten=False,
                        use_sparse=False):
        """
        Returns requested measurement in volumes, returning the default value if
        the hit does not register this measurement

        NOTE: IF COINCIDENCE HASN'T BEEN DEALT WITH THE FIRST HIT ON THE CHANNEL
        WILL BE TAKEN.  The order is determined by the hit_index

        :return: numpy.array of shape [len(events), CDC.n_points]
        """
        # Check if events is empty
        if events is None:
            events = np.arange(self.n_events)
        # Make it an iterable
        _evts = events
        if not _is_sequence(_evts):
            _evts = [_evts]
        # Get the events as an array
        _evts = np.sort(np.array(_evts))
        # Select the relevant event from data
        meas = self.get_events(_evts).groupby([self.event_index,
                                               self.flat_name]).head(1)
        # Return this if we don't want the geometry to come too
        if only_hits:
            return meas[name].values
        # Get the wire_ids and event_ids of the hit data
        wire_ids = meas[self.flat_name].values
        # Map the evnt_ids to the minimal continous set
        evnt_ids = meas.index.droplevel(level=self.hit_index).values
        _, evnt_ids = np.unique(evnt_ids, return_inverse=True)
        # Get the default values
        result = default*np.ones((_evts.size, self.geom.n_points), dtype=dtype)
        # Map this back on to the actual values
        result[evnt_ids, wire_ids] = meas[name].values
        if shift is not None:
            result = result[:, self.geom.shift_wires(shift)]
        if flatten:
            result = result.flatten()
        if use_sparse:
            return result.tocsr()
        return result

    def get_min_layer_events(self, min_layer):
        """
        Returns a boolean series of the events that pass the min_layer criterium
        """
        # Filter for max layer, adding one since counting from zero is less
        # natural here
        max_layer = self.data[self.row_name].groupby(self.event_index).max() + 1
        return max_layer >= 5


    def get_min_hit_events(self, min_hits):
        """
        Returns a boolean series of the events that pass the min_layer criterium
        """
        # Filter for number of signal hits
        return self.data.groupby(self.event_index).size() > min_hits

    def get_track_quality_events(self, min_hits, min_layer):
        """
        Returns a boolean series of the events that pass the min_layer and
        min_hits critereia
        """
        min_layer_events = self.get_min_layer_events(min_layer)
        min_hits_events = self.get_min_hit_events(min_hits)
        return min_layer_events & min_hits_events

    def get_occupancy(self):
        """
        Get the signal occupancy, background occupancy, and total occupancy of
        all events

        :return: np.array (3, self.n_events) as
            (signal_occupauncy, background_occupancy, total_occupancy)
        """
        # Initialize the return value
        occ = np.zeros((3, self.n_events))
        # Get the hit volumes
        hit_vector = self.get_hit_vector()
        # Get the occupancy of each
        occ[0, :] = np.sum(hit_vector == 1, axis=1)
        occ[1, :] = np.sum(hit_vector == 2, axis=1)
        occ[2, :] = np.sum(hit_vector != 0, axis=1)
        return occ

    def print_occupancy(self, occ, verbose=False):
        """
        FIXME
        """
        #TODO documentation
        # Count the number of hits
        evt_index = self.data.index.get_level_values(self.event_index).values
        _, e_n_hits = np.unique(evt_index, return_counts=True)
        # Print the occupancy values
        names = ["n_signal","n_signal_stdev", 
                 "n_background", "n_background_stdev",
                 "n_all", "n_all_stdev",
                 "n_hits", "n_hits_stdev",
                 "n_events_cdc"]
        occ_list = []
        for hits in occ:
            occ_list += [np.average(hits), np.std(hits)]
        # Print the average number of hits
        occ_list += [np.average(e_n_hits), np.std(e_n_hits)]
        # Print the number of events
        occ_list += [self.n_events]
        if verbose:
            print(tabulate.tabulate(zip(names, occ_list)))
        return names, occ_list

class CTHHits(GeomHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, path,
                 geom=CTH(),
                 tree="CTHHitTree",
                 prefix="CTHHit.f",
                 flat_name="VolumeID",
                 chan_name="Channel",
                 idx_name="Counter",
                 trig_name="IsTrig",
                 **kwargs):
        """
        This generates hit data in a structured array from an input root file
        from a file. It assumes the naming convention "M_"+ variable for
        all leaves. It overlays its data on the uses the CTH class to define
        its geometry.

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        :param branches: branches from root file to import
        :param hit_type_name: name of the branch that determines hit type
        :param n_hit_name: name of branch that gives the number of hits in each
                           event
        """
        # Add the flat name as an empty branch
        GeomHits.__init__(self, geom, path, tree, prefix,
                          flat_name=flat_name,
                          **kwargs)
        # Define the row and idx name
        self.data[self.flat_name] = self._get_geom_flat_ids(path, tree,
                                                            prefix+chan_name,
                                                            prefix+idx_name)
        # TODO this removes scintilator light guide hits.  Instead, gain match
        # them to the scintillator outputs
        self.keep_hits_where(variable=self.flat_name,
                             values=self.geom.fiducial_crys)
        # Set the timing column to a
        self.sort_hits(self.time_name)
        # Return the trigger patterns
        self.trig_patterns = None
        self.set_trigger_pattern()
        # Set the name of the trigger column
        self.trig_name = self.prefix + trig_name
        # TODO modulo the signal timing?

    def set_trigger_pattern(self, n_coincidence=2, max_offset=1):
        """
        Set the trigger pattern internally using the required parameters
        """
        # TODO documentation
        self.trig_patterns = \
            self.get_trigger_pattern(n_coincidence=n_coincidence,
                                     max_offset=max_offset)

    def get_trigger_pattern(self, n_coincidence=2, max_offset=1):
        """
        Get the trigger pattern, defaulting to a four fold coincidence with
        a maximal offset of 1 between scintillator and cherenkov pairs
        """
        # TODO documentation
        # Determine all allowed offset volumes
        off_vals = list(range(-max_offset, max_offset + 1))
        # Generate the volume pairs
        vol_pairs = np.array([self.geom.shift_wires(idx)
                              for idx in range(n_coincidence)]).T
        vol_pairs = vol_pairs.astype(np.uint16)
        # Generate the output
        arrays = []
        # Generate the signals
        for c_vols, s_vols in zip([self.geom.up_cher, self.geom.down_cher],
                                  [self.geom.up_scin, self.geom.down_scin]):
            cher_pairs = vol_pairs[c_vols]
            scin_pairs = vol_pairs[s_vols]
            # Roll the pairs to allow for one offset between upper and lower
            # pair matching
            for offset in off_vals:
                arrays += [np.hstack([cher_pairs, np.roll(scin_pairs,
                                                          offset,
                                                          axis=0)])]
        # Provide the trigger sets of data
        return np.vstack(arrays)

    def rebin_time(self, t_bin_ns=10):
        """
        Removes the coincidence by:
            * Summing the energy deposition
            * Taking a signal hit label if it exists on the channel
            * Taking the rest of the values from the earliest hit on the channel
        """
        # TODO documentation
        # Ensure we sum the energy deposition and keep the signal labels
        agg_dict = {branch : 'first' for branch in self.all_branches}
        agg_dict[self.edep_name] = 'sum'
        agg_dict[self.hit_type_name] = 'any'
        # Set the time column to a time series for now
        self.data[self.time_name] = \
            pd.to_datetime(self.data[self.time_name], unit="ns")
        # Group by the event, channels, and rebin to (t_bins_ns*ns) time bins
        time_freq = "{}N".format(t_bin_ns)
        group_list = [self.event_index,
                      self.flat_name,
                      pd.Grouper(key=self.time_name, freq=time_freq)]
        grp_data = self.data.groupby(group_list)
        # Aggregate the data and reset the indexes
        self.data = grp_data.agg(agg_dict)
        self.data.reset_index(level=[1, 2], drop=True, inplace=True)
        # Sort by time first for good measure
        self.data[self.time_name] = \
            pd.to_numeric(self.data[self.time_name], downcast="float")
        self.sort_hits(self.time_name)

    def set_trigger_hits(self, column=None,
                         t_win=50, t_del=10,
                         n_proc=1):
        """
        Set the data column to denote which hits contain the trigger signal
        """
        # TODO documentation
        # Get the default name for the trigger signal labelling
        if column is None:
            column = self.trig_name
        # Prefix the column if needed
        if self.prefix not in column:
            column += self.prefix
        # Make a new boolean array with default false values for each hit
        trigger_data = np.zeros(self.n_hits, dtype=bool)
        trigger_hits = self.find_trigger_hits(t_win=t_win,
                                              t_del=t_del,
                                              n_proc=n_proc)
        # Set the trigger data column
        if trigger_hits is not None:
            trigger_data[trigger_hits] = True
        # Add this column to the data
        self.data[self.trig_name] = trigger_data

    def find_trigger_hits(self, t_win=50, t_del=10, n_proc=1):
        """
        Get the indexes of the hits that make up the trigger signal
        """
        # Reset the index to ensure self.hit_index labels each datum
        # individually
        self._reset_indexes()
        # Set the time column to a time series for now
        # Group by event to find trigger signals in each event
        grp_data = pd.DataFrame(self.data[[self.time_name,
                                           self.flat_name]])
        grp_data[self.time_name] = \
            pd.to_datetime(grp_data[self.time_name], unit="ns")
        grp_data.reset_index(level=self.hit_index, drop=False, inplace=True)
        # Define a lambda function to pass in the arguments we want to use in
        # this iteration of the signal
        trig_scan = partial(self._scan_trigger_windows, t_win=t_win, t_del=t_del)
        # Find the trigger
        trig_hit_idxs = None
        # Use sequential mode if only one cpu requested
        if n_proc == 1:
            grp_data = grp_data.groupby(self.event_index)
            trig_hit_idxs = grp_data.apply(trig_scan)
            trig_hit_idxs = trig_hit_idxs.dropna().values
        # Open up a pool of CPUs otherwise
        else:
            # Using all CPUs if n_proc is none
            if n_proc is None:
                n_proc = mp.cpu_count()
            # Use dask now
            grp_data = dd.from_pandas(grp_data, npartitions=n_proc)
            # Return the values
            with dask.config.set(num_workers=n_proc, scheduler='processes'):
                trig_hit_idxs = grp_data.groupby(self.event_index)
                trig_hit_idxs = trig_hit_idxs.apply(trig_scan, meta=(list))
                trig_hit_idxs = trig_hit_idxs.compute()
            trig_hit_idxs = trig_hit_idxs.dropna().values
        # Return None if there are no hit ids
        if trig_hit_idxs.shape[0] == 0:
            return None
        # Return the correct indexes
        return np.sort(np.concatenate(trig_hit_idxs))

    def _scan_trigger_windows(self, group, t_win=50, t_del=10):
        # TODO documentation
        # Skip if its empty or if its too small
        if group.shape[0] < self.trig_patterns.shape[1]:
            return None
        # Iterate over the ranges
        all_groups = {}
        # Change the offset of the start time for the time rebinning
        for t_start in np.arange(0, t_win, t_del):
            # Get the trigger sets in this group
            trig_set = group.groupby(pd.Grouper(key=self.time_name,
                                                freq=str(t_win)+"N",
                                                base=t_start))
            # Cache them in all groups
            for time, grp in trig_set:
                all_groups[time] = grp
        # Figure out how many time bins overlap
        t_bins = ceil(float(t_win)/t_del)
        # Iterate through all groups in order of their time bin
        sorted_times = list(sorted(all_groups.keys()))
        for idx, time in enumerate(sorted_times):
            # Find any trigger patterns in this time bin
            min_t_hits = self._find_trigger_pattern(all_groups[time])
            # If one is found, check the next t_bins bins to make sure an
            # earlier one does not exist
            if min_t_hits is not None:
                # Scroll through all bins that overlap with the time bin that
                # has the trigger signal
                for next_time in sorted_times[idx+1:idx+t_bins]:
                    # Get the result from the overlapping time bin
                    bin_res = self._find_trigger_pattern(all_groups[next_time])
                    # If this result is earlier than the last, use it instead
                    if (bin_res is not None) and (bin_res[0] < min_t_hits[0]):
                        min_t_hits = bin_res
                # Return the hits from the earliest time
                return min_t_hits[1]
        # Return None by default
        return None

    def _find_trigger_pattern(self, group):
        # TODO document
        # Skip if its empty or if its too small
        if group.shape[0] < self.trig_patterns.shape[1]:
            return None
        # Volume ids and hit times
        vol_ids = group[self.flat_name].values
        # Get if the trigger sets are found in the volume ids
        trig_vols = np.isin(self.trig_patterns, vol_ids)
        # Find the rows where the whole pattern is matched
        trig_ptrns = np.sum(trig_vols, axis=1) == self.trig_patterns.shape[1]
        # Return nothing if nothing found
        if not np.any(trig_ptrns):
            return None
        # Find the minimum time across all patterns
        min_t_hits = (pd.Timestamp.max.to_datetime64(), None)
        hit_times = group[self.time_name].values
        hit_indexes = group[self.hit_index].values
        for pattern in self.trig_patterns[trig_ptrns, :]:
            # Check which hits made the pattern in this group
            hit_ids = np.in1d(vol_ids, pattern)
            # Find their minimum time
            min_time = np.amin(hit_times[hit_ids])
            # Set this to the return value if needed
            if min_time < min_t_hits[0]:
                min_t_hits = (min_time, hit_indexes[hit_ids])
        # Map this back to the ids of the found pattern
        return min_t_hits

    def _get_geom_flat_ids(self, path, tree, chan_name, idx_name):
        """
        Labels each hit by flattened geometry ID to replace the use of volume
        row and volume index
        """
        # Import the data
        branches = [chan_name, idx_name]
        chan_data, idx_data = \
            self._import_root_file(path, tree,
                                   branches=branches)[branches].values.T
        # Map from volume names to row indexes
        row_data = np.vectorize(self.geom.chan_to_row)(chan_data)
        # Flatten the volume names and IDs to flat_voldIDs
        flat_ids = np.zeros(self.n_hits)
        for row, idx, hit in zip(row_data, idx_data, list(range(self.n_hits))):
            flat_ids[hit] = self.geom.point_lookup[row, idx]
        # Save this column and name it
        return flat_ids.astype(np.uint16)

    def get_events(self, events=None, hodoscope="both"):
        """
        Returns the hits from the given event(s).  Default gets all events
        """
        assert hodoscope.startswith("both") or\
               hodoscope.startswith("up") or\
               hodoscope.startswith("down"),\
               "Hodoscope "+ hodoscope +" selected.  This must be both, "+\
               " upstream, or downstream"
        events = super(CTHHits, self).get_events(events)
        if hodoscope.startswith("up"):
            events = self.filter_hits(self.flat_name,
                                      these_hits=events,
                                      values=self.geom.up_crys)
        elif hodoscope.startswith("down"):
            events = self.filter_hits(self.flat_name,
                                      these_hits=events,
                                      values=self.geom.down_crys)
        return events

    def get_trigger_hits(self, events=None):
        """
        Return all of the the trigger hits from the requested events
        """
        # TODO documentation
        # Find the hit indexes of all the volumes that have hits
        return self.filter_hits(self.trig_name,
                                these_hits=self.get_events(events),
                                values=True)

    def get_trigger_time(self, events=None):
        # TODO documentation
        # Get all the trigger hits
        trig_hits = self.get_trigger_hits(events=events)
        # Group by event index and return the minimum
        return trig_hits[self.time_name].groupby(self.event_index).min()

    def get_trigger_hit_idxs(self, events=None):
        """
        Return all of the hit indexes of all trigger hits
        """
        # TODO documentation
        # Get all the trigger hits
        trig_hits = self.get_trigger_hits(events=events)
        # Return the hit index of these hits
        return trig_hits.index.get_level_values(self.hit_index)

    def get_trigger_events(self, events=None, as_bool_series=False):
        """
        FIXME
        """
        # Get all the trigger hits
        event_data = self.get_measurement(events, self.trig_name)
        trig_evts = event_data.groupby(self.event_index).any()
        # Return this if this is all that is needed
        if as_bool_series:
            return trig_evts
        # Otherwise, return the event indexes
        return trig_evts.index[trig_evts].values

    def get_trigger_vector(self, events=None):
        """
        Return the shape [events, self.geom.n_points], where 1 is a triggered
        volume
        """
        # Check if events is none
        if events is None:
            events = np.arange(self.n_events)
        # Allow for a single event
        if not _is_sequence(events):
            events = [events]
        # Find the hit indexes of all the volumes that have hits
        trig_vector = np.zeros((len(events), self.geom.n_points), dtype=bool)
        # Get the trigger hits
        trig_hits = self.get_trigger_hits(events=events)
        vol_indexes = trig_hits[self.flat_name].values
        evt_indexes = trig_hits.index.get_level_values(self.event_index)
        evt_indexes = map_indexes(evt_indexes, np.arange(len(events)))
        # Mark the ones with a trigger as true
        trig_vector[evt_indexes, vol_indexes] = True
        return trig_vector

class CyDetHits():
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, path,
                 cdc_tree="CDCHitTree",
                 cdc_prefix="CDCHit.f",
                 cdc_selection=None,
                 cdc_first_event=0,
                 cdc_n_events=None,
                 cdc_branches=None,
                 cdc_min_time=None,
                 cdc_max_time=None,
                 cdc_drift_time=None,
                 cth_tree="CTHHitTree",
                 cth_prefix="CTHHit.f",
                 cth_selection=None,
                 cth_branches=None,
                 cth_min_time=None,
                 cth_max_time=None,
                 evt_number="EventNumber",
                 hit_number="HitNumber",
                 remove_coincidence=True,
                 **kwargs):
        """
        FIXME
        """
        # TODO documentation
        # Import the CDC sample first
        cdc_hits = CDCHits(path,
                           tree=cdc_tree,
                           prefix=cdc_prefix,
                           selection=cdc_selection,
                           first_event=cdc_first_event,
                           n_events=cdc_n_events,
                           branches=cdc_branches,
                           evt_number=evt_number,
                           hit_number=hit_number,
                           **kwargs)
        # Now import the CTH sample, only importing the events that are already
        # in the CDC sample
        e_idx_vals = cdc_hits.data.index.get_level_values(cdc_hits.event_index)
        e_idx_vals = e_idx_vals.values
        first_evt = np.amin(e_idx_vals)
        last_evt = np.amax(e_idx_vals)
        cth_hit_sel = "({} >= {})".format(cth_prefix+evt_number, first_evt)
        cth_hit_sel += " && ({} <= {})".format(cth_prefix+evt_number, last_evt)
        if cth_selection is None:
            cth_selection = cth_hit_sel
        else:
            cth_selection += " && " + cth_hit_sel
        # Now import the CTH hits and rebin the time
        cth_hits = CTHHits(path,
                           tree=cth_tree,
                           prefix=cth_prefix,
                           selection=cth_selection,
                           branches=cth_branches,
                           evt_number=evt_number,
                           hit_number=hit_number,
                           **kwargs)
        # Set the members
        self.cdc = cdc_hits
        self.cth = cth_hits
        # Check if coincidence should be delt with
        if remove_coincidence:
            # Remove the cdc coincidence
            cdc_hits.remove_coincidence()
            # Rebin the cth hits in time
            cth_hits.rebin_time()
        # Set the event information
        self.event_key = None
        self.reset_event_key()
    
    @property
    def n_events(self):
        return self.event_key.shape[0]

    def reset_event_key(self):
        # TODO documentation
        self.event_key = self._generate_event_key()

    def _generate_event_key(self):
        # TODO documentation
        cdc_events = self.cdc.data.index.get_level_values(self.cdc.event_index)
        cth_events = self.cth.data.index.get_level_values(self.cth.event_index)
        all_events = np.unique(np.concatenate([cdc_events, cth_events]))
        cdc_events = np.isin(all_events, cdc_events)
        cth_events = np.isin(all_events, cth_events)
        event_key = pd.DataFrame({"event_index" : all_events,
                                  "cdc_has_evt" : cdc_events,
                                  "cth_has_evt" : cth_events})
        event_key.set_index("event_index", inplace=True, drop=True)
        return event_key

    @classmethod
    def signal_and_background_sample(cls, sig_path, back_path,
                                     n_proc=None, shuffle_seed=None,
                                     reset_trig=False,
                                     **kwargs):
        # Import the signal
        sig = cls(sig_path, **kwargs)
        # Find and set the trigger hits
        sig.cth.set_trigger_hits(n_proc=n_proc)
        # Figure out which events to keep
        good_trig = sig.cth.get_trigger_events(as_bool_series=True)
        good_trck = sig.cdc.get_track_quality_events(30, 5)
        good_events = good_trig & good_trck
        good_events = good_events[good_events].index
        sig.keep_events(good_events)
        # Import the background and add it to the signal
        sig.add_hits(cls(back_path, **kwargs), shuffle_seed=shuffle_seed)
        # Check if the trigger will be reset
        if reset_trig:
            sig.cth.set_trigger_hits(n_proc=n_proc)
        # Get the the trigger timing
        sig.set_relative_time()
        return sig

    def set_relative_time(self):
        # TODO documentation
        # Get the series of trigger times as an NaN series
        trigger_times = pd.Series(data=np.fill(self.event_key.shape[0], np.nan),
                                  index=self.event_key.index)
        # Get the trigger times for each event
        cth_trigs = self.cth.get_trigger_time()
        trigger_times[cth_trigs.index] = cth_trigs.values
        # Set the relative time for the CDC events
        self.cdc.set_relative_time(trigger_times)

    def add_hits(self, add, trim_self=True, shuffle_seed=None):
        # TODO documentation
        # Complain if there are not enough events
        n_self = self.event_key.shape[0]
        n_add = add.event_key.shape[0]
        if n_self < n_add:
            err_msg = "Not enough events in sample to add new sample.\n"+\
                      "Number of in self {}\n".format(n_self)+\
                      "Number of to add {}".format(n_add)
            raise ValueError(err_msg)
        # Remove the unneeded self events
        self_to_keep = self.event_key.index.values[:n_add]
        self.keep_events(self_to_keep)
        # Map the keys on to eachother
        new_key = map_indexes(add.event_key.index.values,
                              self.event_key.index.values)
        # Loop over the samples
        for geom, mask in zip([add.cdc, add.cth],
                              [add.event_key.cdc_has_evt,
                               add.event_key.cth_has_evt]):
            # Reset the indexes to these new indexes
            geom.set_event_indexes(evts_index=new_key[mask])
        # Reset the event key for the mapped sample
        add.reset_event_key()
        # Add in the sample, reset the event key, and return the object
        self.cdc.add_hits(add.cdc, fix_indexes=False)
        self.cth.add_hits(add.cth, fix_indexes=False)
        self.reset_event_key()

    def print_branches(self):
        """
        Print the names of the data available once you are done
        """
        # Print status message
        print("CTH Branches:")
        self.cth.print_branches()
        print("CDC Branches:")
        self.cdc.print_branches()

    def keep_events(self, events):
        """
        Keep these events in the data
        """
        self.cdc.keep_events(events)
        self.cth.keep_events(events)
        self.event_key = self._generate_event_key()
