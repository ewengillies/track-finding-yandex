"""
Notation used below:
 - wire_id is flat enumerator of all wires
 - layer_id is the index of layer
 - wire_index is the index of wire in the layer
"""
from __future__ import print_function
import numpy as np
from root_numpy import root2array, list_branches
from cylinder import CDC, CTH

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
        branches = [branches]
    return branches

def check_for_branches(path, tree, branches, soft_check=False):
    """
    This checks for the needed branches before they are imported to avoid
    the program to hang without any error messages

    :param path: path to root file
    :param tree: name of tree in root file
    :param branches: required branches
    """
    # Get the names of the availible branches
    availible_branches = list_branches(path, treename=tree)
    # Get the requested branches that are not availible
    bad_branches = list(set(branches) - set(availible_branches))
    # Otherwise, shut it down if its the wrong length
    if bad_branches:
        err_msg = "ERROR: The requested branches:\n"+\
                  "\n".join(bad_branches) + "\n are not availible\n"+\
                  "The branches availible are:\n"+"\n".join(availible_branches)
        if soft_check:
            print(err_msg)
            return False
        # Check that this is zero in length
        assert not bad_branches, err_msg
        # Otherwise return true
    return True

def _add_name_to_branches(path, tree, name, branches, empty_branches):
    """
    Determine which list of branches to put this variable
    """
    # Check if this file already has one
    has_name = check_for_branches(path, tree,
                                  branches=[name],
                                  soft_check=True)
    if has_name:
        branches = _return_branches_as_list(branches)
        branches += [name]
    else:
        empty_branches = _return_branches_as_list(empty_branches)
        empty_branches += [name]
    return branches, empty_branches

class FlatHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self,
                 path,
                 tree='COMETEventsSummary',
                 selection=None,
                 prefix="CDCHit.f",
                 branches=None,
                 empty_branches=None,
                 hit_type_name="IsSig",
                 key_name="EventNumber",
                 use_evt_idx=True, # TODO remove this
                 signal_coding=1,
                 finalize_data=True,
                 n_evts=None):
        """
        Dataset provides an interface to work with MC stored in root format.
        Results of methods are either numpy.arrays or scipy.sparse objects.
        Hits are flattened from [event][evt_hits] structure to [all_hits], with
        look up tables between hits and event stored.

        Additionally, all geometry IDs are flattened from [row][id] to [flat_id]

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        :param branches: branches from root file to import defined for each hit
        :param evt_branches: branches from root file to import defined for each
                             event
        :param hit_type_name: name of the branch that determines hit type
        :param n_hit_name: name of branch that gives the number of hits in each
                           event
        :param signal_coding: value in hit_type_name branch that signifies a
                              signal hit.  Default is 1
        """
        # Assumptions about data naming and signal labelling conventions
        self.prefix = prefix
        self.key_name = self.prefix + key_name
        self.hit_type_name = self.prefix + hit_type_name
        self.signal_coding = signal_coding
        self.n_events = n_evts
        self.use_evt_idx = use_evt_idx
        self.selection = selection
        # Set the number of hits, the number of events, and data to None so that
        # the the next import_root_file knows its the first call
        self.n_hits, self.data = (None, None)

        # Ensure we have a list type of branches
        empty_branches = _return_branches_as_list(empty_branches)
        branches = _return_branches_as_list(branches)
        # Append the prefix if it is not provided
        branches = [self.prefix + branch
                    if not branch.startswith(self.prefix)
                    else branch
                    for branch in branches]
        # Ensure hit type is imported in branches
        branches, empty_branches = _add_name_to_branches(path, tree,
                                                         self.hit_type_name,
                                                         branches,
                                                         empty_branches)

        # Declare out lookup tables
        self.hits_to_events = None
        self.event_to_hits = None
        self.event_to_n_hits = None
        # Get the number of hits for each event
        self._generate_event_to_n_hits_table(path, tree)
        # Fill the look up tables
        self._generate_lookup_tables()

        # Declare the counters
        self.n_hits = None
        self.n_events = None
        # Fill the counters
        self._generate_counters()

        # Get the hit data we want
        if branches:
            data_columns = self._import_root_file(path, tree=tree,
                                               branches=branches)
        # Default to empty list
        else:
            data_columns = []

        # Label each hit with the number of hits its import key
        all_key_column = self._import_root_file(path, tree=tree,
                                                branches=self.key_name)

        # Index each hit by hit ID and by event index
        self.hits_index_name = self.prefix + "hits_index"
        self.event_index_name = self.prefix + "event_index"
        # Index each hit by hit and event
        event_index_column = self.hits_to_events
        hits_index_column = np.arange(self.n_hits)

        # Zip it all together in a record array
        self.all_branches = branches + [self.key_name] +\
                                       [self.hits_index_name] +\
                                       [self.event_index_name]
        self.data = data_columns + all_key_column + \
                    [hits_index_column] + [event_index_column]

        # Add in the empty branches
        empty_branches = _return_branches_as_list(empty_branches)
        # TODO fix hack
        empty_branches = np.unique(empty_branches)
        for branch in empty_branches:
            if branch == self.hit_type_name:
                self.data += [np.zeros(self.n_hits, dtype=bool)]
            else:
                self.data += [np.zeros(self.n_hits)]
            self.all_branches += [branch]

        # Finialize the data if this is the final form
        if finalize_data:
            self._finalize_data()

    def _import_root_file(self, path, tree, branches):
        """
        This wraps root2array to protect the user from importing non-existant
        branches, which cause the program to hang without any error messages

        :param path: path to root file
        :param tree: name of tree in root file
        :param branches: required branches
        """
        # Ensure branches is a list
        if not _is_sequence(branches):
            branches = [branches]
        # Check the branches we want are there
        check_for_branches(path, tree, branches)
        # Grab the branches one by one to save on memory
        data_columns = []
        # TODO absorb event loading limit into selection
        for branch in branches:
            # Grab the branch
            event_data = root2array(path, treename=tree,
                                    branches=[branch],
                                    selection=self.selection)
            # Append the data
            data_columns.append(event_data[branch])
        # Return the columns
        return data_columns

    # TODO depreciate
    def _generate_event_to_n_hits_table(self, path, tree):
        """
        Creates look up tables to map from event index to number of hits from
        the root files
        """
        # Check the branch we need to define the number of hits is there
        check_for_branches(path, tree, branches=[self.key_name])
        # Import the data
        event_data = root2array(path, treename=tree,
                                branches=[self.key_name],
                                selection=self.selection)
        event_data = event_data[self.key_name]
        # Return the number of hits in each event
        if self.use_evt_idx:
            # Check the hits are sorted by event
            assert np.array_equal(event_data, np.sort(event_data)),\
                "Event index named {} not sorted".format(self.key_name)
            # Return the number of hits in each event
            _, event_to_n_hits = np.unique(event_data, return_counts=True)
        else:
            # Assume number of hits per event is stored already
            event_to_n_hits = event_data.copy().astype(int)
        # Trim to the requested number of events
        self.event_to_n_hits = event_to_n_hits[:self.n_events]

    # TODO depreciate
    def _generate_lookup_tables(self):
        """
        Generate mappings between hits and events from current event_to_n_hits
        """
        # Build the look up tables
        first_hit = 0
        try:
            hits_to_events = np.zeros(sum(self.event_to_n_hits))
        except ValueError:
            print(type(self.event_to_n_hits))
        event_to_hits = []
        for event, n_hits in enumerate(self.event_to_n_hits):
            # Record the last hit in the event
            last_hit = first_hit + n_hits
            # Record the range of hit IDs
            event_to_hits.append(np.arange(first_hit, last_hit))
            # Record the event of each hit
            hits_to_events[first_hit:last_hit] = event
            # Shift to the next event
            first_hit = last_hit
        # Shift the event-to-hit list into a numpy object array
        self.event_to_hits = np.array(event_to_hits)
        # Ensure all indexes in hits to events are integers
        self.hits_to_events = hits_to_events.astype(int)

    # TODO depreciate
    def _generate_counters(self):
        """
        Generate the number of events and number of hits
        """
        self.n_hits = len(self.hits_to_events)
        self.n_events = len(self.event_to_n_hits)

    # TODO depreciate
    def _generate_indexes(self):
        '''
        Reset the hit and event indexes
        '''
        self.data[self.hits_index_name] = np.arange(self.n_hits)
        self.data[self.event_index_name] = self.hits_to_events

    # TODO depreciate
    def _reset_all_internal_data(self):
        """
        Reset all look up tables, indexes, and counts
        """
        # Reset the look up tables from the current event_to_n_hits table
        self._generate_lookup_tables()
        # Set the number of hits and events for this data
        self._generate_counters()
        # Set the indexes
        self._generate_indexes()

    # TODO depreciate
    def _reset_event_to_n_hits(self):
        # Find the hits to remove
        self.event_to_n_hits = np.bincount(self.data[self.event_index_name],
                                           minlength=self.n_events)
        # Remove these sums
        assert (self.event_to_n_hits >= 0).all(),\
                "Negative number of hits not allowed!"
        # Trim the look up tables of empty events
        empty_events = np.where(self.event_to_n_hits > 0)[0]
        self._trim_lookup_tables(empty_events)

    def _trim_lookup_tables(self, events):
        """
        Trim the lookup tables to the given event indexes
        """
        # Trim the event indexed tables
        self.event_to_n_hits = self.event_to_n_hits[events]
        # Set the lookup tables
        self._reset_all_internal_data()

    def _finalize_data(self):
        """
        Zip up the data into a rec array if this is the highest level class of
        this instance
        """
        self.data = np.rec.fromarrays(self.data, names=(self.all_branches))
        self._generate_indexes()

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
            this_mask = np.in1d(these_hits[variable], values)
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

    def get_events(self, events=None, unique=True):
        """
        Returns the hits from the given event(s).  Default gets all events

        :param unique: Force each event to only be retrieved once
        """
        # Check if we want all events
        if events is None:
            return self.data
        # Allow for a single event
        if not _is_sequence(events):
            evt_hits = self.event_to_hits[events]
        # Otherwise assume it is a list of events.
        else:
            # Ensure we only get each event once
            # TODO remove and check that this is fine.  Sorts the event ids 
            # before returning them.  This is bad
            if unique:
                events = np.unique(events)
            # Get all the hits we want as flat
            evt_hits = np.concatenate([self.event_to_hits[evt]\
                                       for evt in events])
        # Return the data for these events
        return self.data[evt_hits]

    def trim_events(self, events):
        """
        Keep these events in the data
        """
        self.trim_hits(self.key_name, values=events)

    def sort_hits(self, variable, ascending=True, reset_index=True):
        """
        Sorts the hits by the given variable inside each event.  By default,
        this is done in acending order and the hit index is reset after sorting.
        """
        # Sort each event internally
        for evt in range(self.n_events):
            # Get the hits to sort
            evt_hits = self.event_to_hits[evt]
            # Get the sort order of the given variable
            sort_order = self.data[evt_hits].argsort(order=variable)
            # Reverse the order if required
            if not ascending:
                sort_order = sort_order[::-1]
            # Rearrange the hits
            self.data[evt_hits] = self.data[evt_hits][sort_order]
        # Reset the hit index
        if reset_index:
            self._generate_indexes()

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
        self._reset_event_to_n_hits()

    def add_hits(self, hits, event_indexes=None):
        """
        Append the hits to the current data. If event indexes are supplied, then
        the hits are added event-wise to the event indexes provided.  Otherwise,
        they are stacked on top of each event, starting with event 0
        """
        # TODO fix the fact that flathits will break cause no time_name
        self.data = np.hstack([self.data, hits])
        self.data.sort(order=[self.event_index_name, self.time_name])
        self._reset_event_to_n_hits()

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

    # TODO depreciate
    def get_other_hits(self, hits):
        """
        Returns the hits from the same event(s) as the given hit list
        """
        events = self.hits_to_events[hits]
        events = np.unique(events)
        return self.get_events(events)

    def get_signal_hits(self, events=None):
        """
        Returns the hits from the same event(s) as the given hit list.
        Default gets hits from all events.
        """
        # Get the events
        these_hits = self.filter_hits(self.hit_type_name,
                                      these_hits=self.get_events(events),
                                      values=self.signal_coding)
        return these_hits

    def get_background_hits(self, events=None):
        """
        Returns the hits from the same event(s) as the given hit list
        Default gets hits from all events.
        """
        these_hits = self.filter_hits(self.hit_type_name,
                                      these_hits=self.get_events(events),
                                      values=self.signal_coding,
                                      invert=True)
        return these_hits

    def print_branches(self):
        """
        Print the names of the data available once you are done
        """
        # Print status message
        print("Branches available are:")
        print("\n".join(self.all_branches))

# TODO inheret the MutableSequence attributes of the data directly
#    def __getitem__(self, key)
#    def __setitem__(self, key)
#    def __len__(self, key)

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
                 empty_branches=None,
                 row_name="layerID",
                 idx_name="cellID",
                 edep_name="Charge",
                 time_name="DetectedTime",
                 flat_name="vol_id",
                 trig_name="TrigTime",
                 finalize_data=True,
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
        :param signal_coding: value in hit_type_name branch that signifies a
                              signal hit
        """
        # Name the trigger data row
        self.trig_name = prefix + trig_name
        # Add trig name to branches
        branches, empty_branches = _add_name_to_branches(path,
                                                         tree,
                                                         self.trig_name,
                                                         branches,
                                                         empty_branches)
        # TODO move flat hits to the end
        # Initialize the base class
        FlatHits.__init__(self,
                          path,
                          tree=tree,
                          branches=branches,
                          empty_branches=empty_branches,
                          prefix=prefix,
                          finalize_data=False,
                          **kwargs)

        # Get the geometry flat_IDs
        self.row_name = self.prefix + row_name
        self.idx_name = self.prefix + idx_name
        self.flat_name = self.prefix + flat_name

        # Get the geometry of the detector
        self.geom = geom

        # Build the flattened ID row
        geom_column = self._get_geom_flat_ids(path, tree=tree)

        # Add these data to the data list
        self.data.append(geom_column)
        self.all_branches.append(self.flat_name)

        # Define the names of the time and energy depostition columns
        self.edep_name = self.prefix + edep_name
        self.time_name = self.prefix + time_name

        # Import these, noting this will be ignored if they already exist
        edep_column = self._import_root_file(path, tree=tree,
                                             branches=[self.edep_name])
        time_column = self._import_root_file(path, tree=tree,
                                             branches=[self.time_name])

        # Add these data to the data list
        self.data += edep_column
        self.data += time_column
        self.all_branches.append(self.edep_name)
        self.all_branches.append(self.time_name)

        # Finialize the data if this is the final form
        if finalize_data:
            self._finalize_data()

    def _finalize_data(self):
        """
        Zip up the data into a rec array if this is the highest level class of
        this instance and sort by time
        """
        super(GeomHits, self)._finalize_data()
        self.sort_hits(self.time_name)

    def _get_geom_flat_ids(self, path, tree):
        """
        Labels each hit by flattened geometry ID to replace the use of volume
        row and volume index
        """
        # Import the data
        row_data, idx_data = self._import_root_file(path, tree=tree,
                                                    branches=[self.row_name,
                                                              self.idx_name])
        # Flatten the volume names and IDs to flat_voldIDs
        flat_ids = np.zeros(self.n_hits)
        for row, idx, hit in zip(row_data, idx_data, list(range(self.n_hits))):
            flat_ids[hit] = self.geom.point_lookup[row, idx]
        # Save this column and name it
        flat_id_column = flat_ids.astype(int)
        return flat_id_column

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

    def get_hit_types(self, events, unique=True):
        """
        Returns all hit types, where signal is 1, background is 2,
        nothing is 0.

        :return: numpy.array of shape [CDC.n_points]
        """
        result = np.zeros(self.n_hits, dtype=int)
        # Get the background hits
        bkg_hits = self.get_background_hits(events)[self.hits_index_name]
        result[bkg_hits] = 2
        # Get the signal hits
        sig_hits = self.get_signal_hits(events)[self.hits_index_name]
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
                 chan_name="Channel",
                 finalize_data=True,
                 time_offset=None,
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
        :param signal_coding: value in hit_type_name branch that signifies a
                              signal hit
        """
        # Set the channel name for the flat_ids
        # TODO this is ugly, please clear out the look up table business
        self.chan_name = prefix + chan_name
        # Build the geom hits object
        GeomHits.__init__(self,
                          CDC(),
                          path,
                          tree=tree,
                          prefix=prefix,
                          finalize_data=False,
                          **kwargs)

        # Finialize the data if this is the final form
        if finalize_data:
            self._finalize_data()
        self.time_offset = time_offset

    def _get_geom_flat_ids(self, path, tree):
        """
        Labels each hit by flattened geometry ID to replace the use of volume
        row and volume index
        """
        # Check if chan_name is present is the root file
        has_chan = check_for_branches(path, tree,
                                      branches=[self.chan_name],
                                      soft_check=True)
        if has_chan:
            return self._import_root_file(path, tree,
                                          branches=[self.chan_name])[0]
        return super(CDCHits, self)._get_geom_flat_ids(path, tree)


    def get_measurement(self, name, events=None, shift=None, default=0,
                        only_hits=True, flatten=False, use_sparse=False):
        """
        Returns requested measurement in volumes, returning the default value if
        the hit does not register this measurement

        NOTE: IF COINCIDENCE HASN'T BEEN DEALT WITH THE FIRST HIT ON THE CHANNEL
        WILL BE TAKEN.  The order is determined by the hits_index

        :return: numpy.array of shape [len(events), CDC.n_points]
        """
        # Check if events is empty
        if events is None:
            events = np.arange(self.n_events)
        # Make it an iterable
        my_events = events
        if not _is_sequence(my_events):
            my_events = [my_events]
        # Get the events as an array
        my_events = np.sort(np.array(my_events))
        # Select the relevant event from data
        meas = self.get_events(my_events)[name]
        # Get the wire_ids and event_ids of the hit data
        wire_ids = self.get_hit_vols(my_events, unique=False)
        # Map the evnt_ids to the minimal continous set
        evnt_ids = np.repeat(np.arange(my_events.size),
                             self.event_to_n_hits[my_events])
        # Get the default values
        result = default*np.ones((my_events.size, self.geom.n_points),
                                  dtype=float)
        # Figure out the first place each wire id is mentioned in each event
        two_d_ids = np.vstack([evnt_ids, wire_ids]).T
        # Get the unique evt_id/wire_id values
        one_d_view = np.ascontiguousarray(two_d_ids).view(
            np.dtype((np.void, two_d_ids.dtype.itemsize * two_d_ids.shape[1])))
        # Figure out where they appear
        _, first_hits = np.unique(one_d_view, return_index=True)
        # Map their appearances back in the correct order
        evnt_ids, wire_ids = two_d_ids[np.sort(first_hits)].T
        meas = meas[np.sort(first_hits)]
        # Put the result on the correct wires
        result[evnt_ids, wire_ids] = meas
        if shift is not None:
            result = result[:, self.geom.shift_wires(shift)]
        if only_hits:
            result = result[evnt_ids, wire_ids]
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
        return np.unique(self.get_events(events=good_event_idx)[self.key_name])

    def min_hits_cut(self, min_hits):
        """
        Returns the EventNumbers of the events that pass the min_hits criterium
        """
        # Filter for number of signal hits
        good_events = np.where(self.event_to_n_hits >= min_hits)[0]
        # TODO these should not return key name, this is dangeous for hit
        # merging
        return np.unique(self.get_events(events=good_events)[self.key_name])

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
                 finalize_data=True,
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
        :param signal_coding: value in hit_type_name branch that signifies a
                              signal hit
        """
        GeomHits.__init__(self,
                          CTH(),
                          path,
                          tree=tree,
                          prefix=prefix,
                          row_name=row_name,
                          idx_name=idx_name,
                          time_name=time_name,
                          finalize_data=False,
                          **kwargs)

        if finalize_data:
            self._finalize_data()

    def _finalize_data(self):
        """
        Zip up the data into a rec array if this is the highest level class of
        this instance and sort by time
        """
        super(CTHHits, self)._finalize_data()
        # Remove passive volumes from the hit data
        # TODO fix passive volume hack
        self.trim_hits(variable=self.flat_name, values=self.geom.fiducial_crys)

    def _get_geom_flat_ids(self, path, tree):
        """
        Labels each hit by flattened geometry ID to replace the use of volume
        row and volume index
        """
        # Import the data
        chan_data = self._import_root_file(path, tree=tree,
                                           branches=[self.row_name])[0]
        idx_data = self._import_root_file(path, tree=tree,
                                          branches=[self.idx_name])[0]
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
        events = super(self.__class__, self).get_events(events)
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
            trig_hits = self.filter_hits(self.flat_name,
                                         these_hits=self.get_events(event),
                                         values=trig_vols)[self.hits_index_name]
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
                                invert=True)[self.hits_index_name]

    def get_trig_evts(self, events=None):
        """
        Return the trigger events by EventNumber
        """
        # Find the hit indexes of all the volumes that have hits
        # TODO have this not return keyname, this is dangerous for hit merging
        return np.unique(self.data[self.get_trig_hits(events)][self.key_name])

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
        shared_evts = np.intersect1d(self.cdc.get_events()[self.cdc.key_name],
                                     self.cth.get_events()[self.cth.key_name])
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
