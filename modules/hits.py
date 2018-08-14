"""
Notation used below:
 - wire_id is flat enumerator of all wires
 - layer_id is the index of layer
 - wire_index is the index of wire in the layer
"""
from __future__ import print_function
import numpy as np
from cylinder import CDC, CTH
from uproot_selected import import_uproot_selected, check_for_branches

# TODO swith to pandas
    # TODO move data_tools into hits
    # TODO improve CTH tigger logic to have time window
    # TODO improve CTH hits to sum over 10ns bins
    # TODO deal with multi-indexing events from evt_number
        # TODO deal with empty CTH events or empty CDC events:
        #      sparse DF with both CTH and CDC hits.
        #      * do messy version for now, just need:
        #         * set the CTH logic to work in this
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
    return np.array([map_v[x] for x in v_old])[inv_old]

class FlatHits():
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self,
                 path,
                 tree='COMETEventsSummary',
                 selection=None,
                 first_event=0,
                 n_events=None,
                 prefix="CDCHit.f",
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
        # Ensure the branch names are unique
        self._branches = sorted(list(set(self._branches)))
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

    def get_events(self, events=None):
        """
        Returns the hits from the given event(s).  Default gets all events

        :param unique: Force each event to only be retrieved once
        """
        # Return the data for these events
        if events is None:
            return self.data
        # Return all events by default
        loc_lvl = self.data.index.names.index(self.event_index)
        loc_val = self.data.index.levels[loc_lvl][events]
        return self.data.loc[loc_val]

    def trim_events(self, events):
        """
        Keep these events in the data
        """
        self.trim_hits(self.evt_number, values=events)

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

    def trim_hits(self, variable, values=None, greater_than=None,
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
    def __init__(self,
                 geom,
                 path,
                 tree='COMETEventsSummary',
                 prefix="CDCHit.f",
                 branches=None,
                 edep_name="Charge",
                 time_name="DetectedTime",
                 flat_name="vol_id",
                 trig_name="TrigTime",
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
        # Name the trigger data row
        self.trig_name = prefix + trig_name
        # Define the names of the time and energy depostition columns
        branches = _return_branches_as_list(branches)
        self.edep_name = prefix + edep_name
        self.time_name = prefix + time_name
        branches += [self.edep_name, self.time_name]
        # Define the flattened indexes of the geometry
        self.flat_name = prefix + flat_name
        # Get the geometry of the detector
        self.geom = geom
        # Initialize the base class
        FlatHits.__init__(self,
                          path,
                          tree=tree,
                          branches=branches,
                          prefix=prefix,
                          **kwargs)

    def get_measurement(self, events, name):
        """
        Returns requested measurement by event

        :return: numpy.array of length self.n_hits
        """
        # Select the relevant event from data
        return self.get_events(events)[name]

    def get_hit_vols(self, events, unique=True, hit_type="both"):
        """
        Returns the sequence of flat_ids that register hits in given event

        :return: numpy array of hit wires
        :param: hit_type defines which hit volumes should be retrieved.
                Possible valuses are both, signal, and background
        """
        # Select the relevant event from data
        hit_type = hit_type.lower()
        assert hit_type.startswith("both") or\
               hit_type.startswith("sig") or\
               hit_type.startswith("back"),\
               "Hit type "+ hit_type+ " selected.  This must be both, signal,"+\
               " or background"
        if hit_type == "both":
            flat_ids = self.get_events(events)[self.flat_name]
        elif hit_type.startswith("sig"):
            flat_ids = self.get_signal_hits(events)[self.flat_name]
        elif hit_type.startswith("back"):
            flat_ids = self.get_background_hits(events)[self.flat_name]
        if unique is True:
            flat_ids = np.unique(flat_ids)
        return flat_ids

    def get_sig_vols(self, events, unique=True):
        """
        Returns the sequence of flat_ids that register signal hits in given
        event

        :return: numpy array of hit wires
        """
        # Select the relevant event from data
        return self.get_hit_vols(events, unique, hit_type="sig")

    def get_bkg_vols(self, events, unique=True):
        """
        Returns the sequence of flat_ids that register hits in given event

        :return: numpy array of hit wires
        """
        # Select the relevant event from data
        return self.get_hit_vols(events, unique, hit_type="back")

    def get_hit_vector(self, events):
        """
        Returns a vector denoting whether or not a wire has a hit on it. Returns
        1 for a hit, 0 for no hit

        :return: numpy array of shape [n_wires] whose value is 1 for a hit, 0
                 for no hit
        """
        # Get the flat vol IDs for those with hits
        hit_vols = self.get_hit_vols(events, unique=True)
        # Make the hit vector
        hit_vector = np.zeros(self.geom.n_points)
        hit_vector[hit_vols] = 1
        return hit_vector

    def get_vol_types(self, events):
        """
        Get hits in all volumes by type, 1 is signal, 2 in background, nothing
        is 0. Signal takes priority.

        :return: numpy.array of shape [Geometry.n_points]
        """
        # Get the flat vol IDs for those vols with sig or bkg hits
        bkg_vols = self.get_bkg_vols(events, unique=True)
        sig_vols = self.get_sig_vols(events, unique=True)
        # Make the hit vector
        hit_vector = np.zeros(self.geom.n_points)
        hit_vector[bkg_vols] = 2
        hit_vector[sig_vols] = 1
        return hit_vector

    def get_hit_types(self, events):
        """
        Returns all hit types, where signal is 1, background is 2,
        nothing is 0.

        :return: numpy.array of shape [CDC.n_points]
        """
        result = np.zeros(self.n_hits, dtype=int)
        # Get the background hits
        bkg_hits = self.get_background_hits(events)[self.hit_index]
        result[bkg_hits] = 2
        # Get the signal hits
        sig_hits = self.get_signal_hits(events)[self.hit_index]
        result[sig_hits] = 1
        return result.astype(int)

    def get_energy_deposits(self, events):
        """
        Returns energy deposit in all wires

        :return: numpy.array of shape [CDC.n_points]
        """
        energy_deposit = self.get_measurement(events, self.edep_name)
        return energy_deposit

    def get_hit_time(self, events):
        """
        Returns the timing of the hit

        :return: numpy.array of shape [CDC.n_points]
        """
        time_hit = self.get_measurement(events, self.time_name)
        return time_hit

    def get_trigger_time(self, events):
        """
        Returns the timing of the trigger on an event

        :return: numpy.array of shape [CDC.n_points]
        """
        # Check the trigger time has been set
        assert "CDCHit.fTrigTime" in self.all_branches,\
                "Trigger time has not been set yet"
        return self.get_measurement(events, self.trig_name)

    def get_relative_time(self, events):
        """
        Returns the difference between the start time of the hit and the time of
        the trigger.
        """
        trig_time = self.get_trigger_time(events)
        hit_time = self.get_hit_time(events)
        return hit_time - trig_time


class CDCHits(GeomHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self,
                 path,
                 tree='COMETEventsSummary',
                 prefix="CDCHit.f",
                 branches=None,
                 flat_name="Channel",
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
        # Build the geom hits object
        GeomHits.__init__(self,
                          CDC(),
                          path,
                          tree=tree,
                          branches=branches,
                          prefix=prefix,
                          flat_name=flat_name,
                          **kwargs)
        # Sort the hits by time
        self.sort_hits(self.time_name)

    def remove_coincidence(self, sort_hits=True):
        """
        Removes the coincidence by:
            * Summing the energy deposition
            * Taking a signal hit label if it exists on the channel
            * Taking the rest of the values from the earliest hit on the channel
        """
        # Sort the hits by channel and by time
        self.sort_hits([self.flat_name, self.time_name])
        # Group by the channels
        chan_groups = self.data.groupby([self.event_index,
                                         self.flat_name])
        # Ensure we sum the energy deposition and keep the signal labels
        agg_dict = {}
        agg_dict[self.edep_name] = 'sum'
        agg_dict[self.hit_type_name] = 'any'
        agg_dict[self.flat_name] = 'mean'
        # Evaluate the groups for these two special cases
        _cached_vals = chan_groups.agg(agg_dict)
        # Get the first element for the rest of the data
        self.data = chan_groups.head(1)
        # Ensure the channel number match along all events before setting the
        # special information, as an extra precaution
        err = "Improperly trying to set edep and sig label"
        np.testing.assert_allclose(_cached_vals.loc[:, (self.flat_name)],
                                   self.data.loc[:, (self.flat_name)],
                                   err_msg=err)
        # Reset the special data
        self.data.loc[:, (self.edep_name, self.hit_type_name)] =\
            _cached_vals.loc[:, (self.edep_name, self.hit_type_name)].values
        # Sort the hits by time again
        # TODO remove once over
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

    def get_hit_types(self, events, unique=True):
        """
        Returns hit type in all volumes, where signal is 1, background is 2,
        nothing is 0.  If signal and background are both incident, signal takes
        priority

        :return: numpy.array of shape [CDC.n_points]
        """
        result = np.zeros(self.geom.n_points, dtype=int)
        # Get the background hits
        bkg_hits = np.unique(self.get_background_hits(events)[self.flat_name])
        result[bkg_hits] = 2
        # Get the signal hits
        sig_hits = np.unique(self.get_signal_hits(events)[self.flat_name])
        result[sig_hits] = 1
        return result.astype(int)

    def get_hit_wires_even_odd(self, events):
        """
        Returns two sequences of wire_ids that register hits in given event, the
        first is only in even layers, the second is only in odd layers

        :return: numpy array of hit wires
        """
        hit_wires = self.get_hit_vols(events)
        odd_wires = np.where((self.geom.point_pol == 1))[0]
        even_hit_wires = np.setdiff1d(hit_wires, odd_wires, assume_unique=True)
        odd_hit_wires = np.intersect1d(hit_wires, odd_wires, assume_unique=True)
        return even_hit_wires, odd_hit_wires

    def get_hit_vector_even_odd(self, events):
        """
        Returns a vector denoting whether or not a wire on an odd layer has a
        hit on it. Returns 1 for a hit in an odd layer, 0 for no hit and all
        even layers

        :return: numpy array of shape [n_wires] whose value is 1 for a hit on an
                odd layer, 0 otherwise
        """
        even_wires, odd_wires = self.get_hit_wires_even_odd(events)
        even_hit_vector = np.zeros(self.geom.n_points)
        even_hit_vector[even_wires] = 1
        odd_hit_vector = np.zeros(self.geom.n_points)
        odd_hit_vector[odd_wires] = 1
        return even_hit_vector, odd_hit_vector

    def min_layer_cut(self, min_layer):
        """
        Returns the EventNumbers of the events that pass the min_layer criterium
        """
        # Filter for max layer
        evt_max = np.zeros(self.n_events)
        for event in range(self.n_events):
            evt_hits = self.get_measurement(self.flat_name,
                                            events=event,
                                            flatten=False,
                                            only_hits=False).astype(int)
            evt_max[event] = np.amax(self.geom.point_layers[evt_hits])
        good_event_idx = np.where(evt_max >= min_layer)
        # TODO these should not return key name, this is dangeous for hit
        # merging
        return np.unique(self.get_events(events=good_event_idx)[self.evt_number])

    def min_hits_cut(self, min_hits):
        """
        Returns the EventNumbers of the events that pass the min_hits criterium
        """
        # Filter for number of signal hits
        good_events = np.where(self.event_to_n_hits >= min_hits)[0]
        # TODO these should not return key name, this is dangeous for hit
        # merging
        return np.unique(self.get_events(events=good_events)[self.evt_number])

    def get_occupancy(self):
        """
        Get the signal occupancy, background occupancy, and total occupancy of
        all events

        :return: np.array (3, self.n_events) as
            (signal_occupauncy, background_occupancy, total_occupancy)
        """
        occ = np.zeros((3, self.n_events))
        for evt in range(self.n_events):
            occ[0, evt] = len(self.get_sig_vols(evt))
            occ[1, evt] = len(self.get_bkg_vols(evt))
            occ[2, evt] = len(self.get_hit_vols(evt))

        # print some information
        avg_n_hits, err_n_hits = np.average(self.event_to_n_hits), \
                           np.std(self.event_to_n_hits)/np.sqrt(self.n_events)
        sig_occ, sig_err = np.average(occ[0, :]), \
                           np.std(occ[0, :])/np.sqrt(self.n_events)
        back_occ, back_err = np.average(occ[1, :]), \
                           np.std(occ[1, :])/np.sqrt(self.n_events)
        all_occ, all_err = np.average(occ[2, :]), \
                           np.std(occ[2, :])/np.sqrt(self.n_events)
        print("Sig Occ: {} {}".format(sig_occ, sig_err))
        print("Back Occ: {} {}".format(back_occ, back_err))
        print("All Occ: {} {}".format(all_occ, all_err))
        print("NumHits: {} {}".format(avg_n_hits, err_n_hits))
        print("MinMultiHit: {}".format((avg_n_hits - all_occ)/float(all_occ)))
        return occ

class CTHHits(GeomHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self,
                 path,
                 tree='COMETEventsSummary',
                 prefix="CTHHit.f",
                 row_name="Channel",
                 idx_name="Counter",
                 time_name="MCPos.fE",
                 flat_name="vol_id",
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
        GeomHits.__init__(self,
                          CTH(),
                          path,
                          tree=tree,
                          prefix=prefix,
                          flat_name=flat_name,
                          time_name=time_name,
                          **kwargs)
        # Define the row and idx name
        flat_ids = self._get_geom_flat_ids(path, tree,
                                           prefix+row_name,
                                           prefix+idx_name)
        self.data[self.flat_name] = flat_ids.astype(np.uint16)
        # TODO check if we should trim this or not
        self.trim_hits(variable=self.flat_name, values=self.geom.fiducial_crys)
        self.sort_hits(self.time_name)

    def _get_geom_flat_ids(self, path, tree, row_name, idx_name):
        """
        Labels each hit by flattened geometry ID to replace the use of volume
        row and volume index
        """
        # Import the data
        branches = [row_name, idx_name]
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
        flat_id_column = flat_ids.astype(int)
        return flat_id_column

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

    def _find_trigger_signal(self, vol_types):
        """
        Returns the volumes that take part in the trigger pattern given an array
        of volume types

        :param vol_types: np.array of shape [self.geom.n_points] whose value is
                          non-zero for a volume hit
        :return trig_vols: np.array of shape [self.geom.n_points] whose value is
                           1 for all volumes that form a trigger shape
        """
        # Get all volumes with pairs
        hit_and_left = np.logical_and(vol_types,
                                      vol_types[self.geom.shift_wires(1)])
        hit_and_right = np.logical_and(vol_types,
                                       vol_types[self.geom.shift_wires(-1)])
        trig_crys = np.logical_or(hit_and_left, hit_and_right)
        # Get volumes with crystals hit above or below
        on_top = np.logical_and(trig_crys[self.geom.cher_crys],
                                trig_crys[self.geom.scin_crys])
        trig_crys[self.geom.cher_crys] = on_top
        trig_crys[self.geom.scin_crys] = on_top
        # Include the crystals to the left and right of these volumes
        trig_crys = np.logical_or.reduce((trig_crys,
                                          trig_crys[self.geom.shift_wires(1)],
                                          trig_crys[self.geom.shift_wires(-1)]))
        # Return the volumes that pass and have hits
        return np.logical_and(vol_types, trig_crys)


    def set_trigger_time(self):
        # Sort by time first
        self.sort_hits(self.time_name)
        # Reset the trigger timing
        self.data[self.trig_name] = 0
        for event in range(self.n_events):
            # Get the volumes with hits for these events
            vol_types = self.get_vol_types(event)
            trig_vols = np.nonzero(self._find_trigger_signal(vol_types))[0]
            # Skip the event if there is no trigger
            if len(trig_vols) == 0:
                continue
            # Find the hit indexes of all the volumes that have hits
            # TODO change for dataframe
            trig_hits = self.filter_hits(self.flat_name,
                                         these_hits=self.get_events(event),
                                         values=trig_vols)[self.hit_index]
            # Get the indexes where these volumes first appear in the
            # (event,time) sorted data
            _, uniq_idxs = np.unique(self.data[self.flat_name][trig_hits],
                                     return_index=True)
            # Get as close to the fourth hit as possible, but not less
            uniq_idxs = uniq_idxs[uniq_idxs > 2]
            try:
                fourth_uniq_hit = uniq_idxs[(np.abs(uniq_idxs-3)).argmin()]
                fourth_vol_hit = trig_hits[fourth_uniq_hit]
            except:
                print("Error in trigger logic!!\n"+\
                      "Trig vols : {}\n".format(trig_vols)+\
                      "Uniq idx : {}\n".format(uniq_idxs)+\
                      "Event : {}".format(event))
            self.data[self.trig_name][trig_hits] = \
                    self.data[self.time_name][fourth_vol_hit]

# TODO move these into get measurement

    def get_trig_hits(self, events=None):
        """
        Return the trigger hit hit_index in the given event
        """
        # Find the hit indexes of all the volumes that have hits
        return self.filter_hits(self.trig_name,
                                these_hits=self.get_events(events),
                                values=0,
                                invert=True)[self.hit_index]

    def get_trig_evts(self, events=None):
        """
        Return the trigger events by EventNumber
        """
        # Find the hit indexes of all the volumes that have hits
        # TODO have this not return keyname, this is dangerous for hit merging
        return np.unique(self.data[self.get_trig_hits(events)][self.evt_number])

    def get_trig_vector(self, events):
        """
        Return the shape [events, self.geom.n_points], where 1 is a triggered
        volume
        """
        # Allow for a single event
        if not _is_sequence(events):
            events = [events]
        # Find the hit indexes of all the volumes that have hits
        trig_vector = np.zeros((len(events), self.geom.n_points))
        for index, evt in enumerate(events):
            trig_vols = self.data[self.flat_name][self.get_trig_hits(evt)]
            trig_vols = np.unique(trig_vols)
            trig_vector[index, trig_vols] = 1
        return trig_vector[:, self.geom.fiducial_crys]

class CyDetHits(FlatHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self,
                 cdc_hits,
                 cth_hits,
                 common_events=False):
        """
        A class to support overlaying hit classes of the same type.  This will
        returned the combined event from each of the underlying hit classes.
        """
        # TODO assertion here
        self.cth = cth_hits
        self.cdc = cdc_hits
        if common_events:
            self.keep_common_events()
        self.n_events = min(self.cdc.n_events, self.cth.n_events)

    def keep_common_events(self):
        """
        Trim all events by event index so that they have the same events
        """
        shared_evts = np.intersect1d(self.cdc.get_events()[self.cdc.evt_number],
                                     self.cth.get_events()[self.cth.evt_number])
        self.trim_events(shared_evts)

    def set_trigger_time(self):
        """
        Set the CTH trigger time for both the CTH and for all CDC hits
        """
        # Set the CTH trigger time
        self.cth.set_trigger_time()
        # Broadcase this value to all CDC hits in the event
        for event in range(self.cth.n_events):
            # Get all the trigger times in  this event
            all_trig = np.unique(self.cth.get_events(event)[self.cth.trig_name])
            # Remove the zero values
            evt_trig = np.trim_zeros(all_trig)
            # Broadcast this value to all CDC hits in this event
            self.cdc.data[self.cdc.trig_name][self.cdc.event_to_hits[event]] = \
                evt_trig

    def print_branches(self):
        """
        Print the names of the data available once you are done
        """
        # Print status message
        print("CTH Branches:")
        self.cth.print_branches()
        print("CDC Branches:")
        self.cdc.print_branches()

    def trim_events(self, events):
        """
        Keep these events in the data
        """
        self.cdc.trim_events(events)
        self.cth.trim_events(events)
        self.n_events = self.cdc.n_events

    def apply_timing_cut(self, lower=700, upper=1170, drift=450):
        """
        Remove the hits that do not pass timing cut
        """
        self.cth.trim_hits(variable=self.cth.time_name,\
                           greater_than=lower, less_than=upper)
        self.cdc.trim_hits(variable=self.cdc.time_name,\
                           greater_than=lower, less_than=upper+drift)
