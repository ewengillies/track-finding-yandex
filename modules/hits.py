import numpy as np
import numpy.lib.recfunctions
from root_numpy import root2array
from cylinder import CyDet
from random import Random
from scipy.sparse import lil_matrix, find

"""
Notation used below:
 - wire_id is flat enumerator of all wires
 - layer_id is the index of layer
 - wire_index is the index of wire in the layer
"""

class FlatHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self, path="../data/151208_SimChen_noise.root",
                 tree='tree', prefix="CdcCell", branches=None,
                 hit_type_name="hittype", n_hits_name="nHits",
                 signal_coding=1, build_record=True, finalize_data=True):
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
        self.prefix = prefix + "_"
        self.n_hits_name = self.prefix + n_hits_name
        self.hit_type_name = self.prefix + hit_type_name
        self.signal_coding = signal_coding

        # Deal with requested branches
        # Ensure branches are given as list
        if branches is None:
            branches = []
        if not isinstance(branches, list):
            branches = [branches]
        # Append the prefix if it is not provided
        branches = [self.prefix + branch
                    if not branch.startswith(self.prefix)
                    else branch
                    for branch in branches]
        # Ensure hit type is imported in branches
        if self.hit_type_name not in branches:
            branches += [self.hit_type_name]

        # Set the number of hits, the number of events, and data to None so that
        # the the next import_root_file knows its the first call
        self.n_hits, self.n_events, self.data = (None, None, None)

        # Initialize our data and look up tables
        self.hits_to_events, self.event_to_hits, self.event_to_n_hits =\
            self._get_event_to_hits_lookup(path, tree=tree)

        # Set the number of hits and events for this data
        self.n_hits = len(self.hits_to_events)
        self.n_events = len(self.event_to_n_hits)

        # Get the hit data we want
        if not branches is None:
            data_columns = self._import_root_file(path, tree=tree,
                                               branches=branches)
        # Default to empty list
        else:
            data_columns = []

        # Label each hit with the number of hits in its event
        all_n_hits_column = [self.event_to_n_hits[self.hits_to_events]]

        # Index each hit
        self.hits_index_name = self.prefix + "hits_index"
        hits_index_column = [np.arange(self.n_hits)]

        # Zip it all together in a record array
        self.all_branches = branches + [self.n_hits_name] +\
                                       [self.hits_index_name]
        self.data = data_columns + all_n_hits_column + hits_index_column

        # Finialize the data if this is the final form
        if finalize_data:
            self._finalize_data()

    def _finalize_data(self):
        """
        Zip up the data into a rec array if this is the highest level class of
        this instance
        """
        self.data = np.rec.fromarrays(self.data, names=(self.all_branches))

    def print_branches(self):
        """
        Print the names of the data available once you are done
        """
        # Print status message
        print "Branches available are:"
        print "\n".join(self.all_branches)

    def _check_for_branches(self, path, tree, branches):
        """
        This checks for the needed branches before they are imported to avoid
        the program to hang without any error messages

        :param path: path to root file
        :param tree: name of tree in root file
        :param branches: required branches
        """
        # Import one event with all the branches to get the names of the
        # branches
        dummy_root = root2array(path, treename=tree, start=0, stop=1)
        # Get the names of the imported branches
        availible_branches = dummy_root.dtype.names
        # Get the requested branches that are not availible
        bad_branches = list(set(branches) - set(availible_branches))
        # Check that this is zero in length
        assert len(bad_branches) == 0, "ERROR: The requested branches:\n"+\
                "\n".join(bad_branches) + "\n are not availible"

    def _import_root_file(self, path, tree, branches):
        """
        This wraps root2array to protect the user from importing non-existant
        branches, which cause the program to hang without any error messages

        :param path: path to root file
        :param tree: name of tree in root file
        :param branches: required branches
        """
        # Ensure branches is a list
        if not isinstance(branches, list):
            branches = [branches]
        # Check that these branches do not already exist locally
        if self.data is not None:
            get_branches = list(set(branches) - set(self.all_branches))
            had_branches = list(set(branches) - set(get_branches))
            # Print a warning if branches have been requested multiple times
            had_branches = "\n".join(had_branches)
            if len(had_branches) != 0:
                print "Warning : Branches " + had_branches + " \n already "+\
                      "exist locally.  These will not be imported again."
        else:
            get_branches = branches
        # Check the braches we want are there
        self._check_for_branches(path, tree, get_branches)
        # Grab the branches one by one to save on memory
        data_columns = []
        for branch in get_branches:
            # Grab the branch
            event_data = root2array(path, treename=tree,\
                                    branches=[branch])
            # If we know the number of hits and events, require the branch is as
            # long as one of these
            if (self.n_hits is not None) and (self.n_events is not None):
                # Concatonate the branch if it is an array of lists, i.e. if it
                # is defined for every hit
                if event_data.dtype[branch] == object:
                    event_data = np.concatenate(event_data[branch])
                    # Check that the right number of hits are defined
                    data_length = len(event_data)
                # Otherwise assume it is defined event-wise, stretch it by event
                # so each hit has the value corresponding to its event.
                else:
                    # Check the length
                    data_length = len(event_data)
                    event_data = event_data[branch][self.hits_to_events]
                # Check that the length of the array makes sense
                assert (data_length == self.n_hits) or\
                       (data_length == self.n_events),\
                       "ERROR: The length of the data in the requested "+\
                       "branch " + branch + " is not the length of the "+\
                       "number of events or the number of hits"
                # Add this branch
                data_columns.append(event_data)
            # If we do not know the number of hits and events, assume its
            # defined hit-wise
            else:
                data_columns.append(np.concatenate(event_data[branch]))
        # Return
        return data_columns



    def _get_event_to_hits_lookup(self, path, tree):
        """
        Creates look up tables to map from events to hits index and from
        hit to event number
        """
        # Check the branch we need to define the number of hits is there
        self._check_for_branches(path, tree, branches=[self.n_hits_name])
        # Import the data
        event_data = root2array(path, treename=tree,
                                branches=[self.n_hits_name])
        # Initialize a look up table that maps from hit number to event number
        hits_to_events = np.zeros(sum(event_data[self.n_hits_name]))
        # Create a look up table that maps from event number the range of hits
        # IDs in that event
        event_to_hits = []
        # Store the number of hits in each event
        event_to_n_hits = event_data[self.n_hits_name].copy().astype(int)
        # Build the look up tables
        first_hit = 0
        for event in range(len(event_data)):
            # Record the last hit in the event
            last_hit = first_hit + event_to_n_hits[event]
            # Record the range of hit IDs
            event_to_hits.append(np.arange(first_hit, last_hit))
            # Record the event of each hit
            hits_to_events[first_hit:last_hit] = event
            # Shift to the next event
            first_hit = last_hit
        # Shift the event-to-hit list into a numpy object array
        event_to_hits = np.array(event_to_hits)
        # Ensure all indexes in hits to events are integers
        hits_to_events = hits_to_events.astype(int)

        # Return the lookup tables
        return hits_to_events, event_to_hits, event_to_n_hits

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
            sort_order = self.data[evt_hits][variable].argsort()
            # Reverse the order if required
            if ascending == False:
                sort_order = sort_order[::-1]
            # Rearrange the hits
            self.data[evt_hits] = self.data[evt_hits][sort_order]
        # Reset the hit index
        if reset_index == True:
            self.data[self.hits_index_name] = np.arange(self.n_hits)

    def get_events(self, events=None, unique=True):
        """
        Returns the hits from the given event(s).  Default gets all events

        :param unique: Force each event to only be retrieved once
        """
        # Check if we want all events
        if events is None:
            return self.data
        # Allow for a single event
        if isinstance(events, int):
            evt_hits = self.event_to_hits[events]
        # Otherwise assume it is a list of events.
        else:
            # Ensure we only get each event once
            if unique:
                events = np.unique(events)
            # Get all the hits we want as flat
            evt_hits = np.concatenate([self.event_to_hits[evt]\
                                       for evt in events])
        # Return the data for these events
        return self.data[evt_hits]

    def filter_hits(self, selected_hits, variable, values, invert=False):
        """
        Returns the section of the data where the variable equals
        any of the values
        """
        # Switch to a list if a single value is given
        if not isinstance(values, list):
            values = [values]
        mask = np.in1d(selected_hits[variable], values, invert=invert)
        return selected_hits[mask]

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
        these_hits = self.get_events(events)
        these_hits = self.filter_hits(these_hits, self.hit_type_name,
                                      self.signal_coding)
        return these_hits

    def get_background_hits(self, events=None):
        """
        Returns the hits from the same event(s) as the given hit list
        Default gets hits from all events.
        """
        these_hits = self.get_events(events)
        these_hits = self.filter_hits(these_hits, self.hit_type_name,
                                      self.signal_coding, invert=True)
        return these_hits

class GeomHits(FlatHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    # pylint: disable=unbalanced-tuple-unpacking
    def __init__(self, path="../data/signal.root", tree='tree',
                 branches=None, prefix="CdcCell", hit_type_name="hittype",
                 n_hits_name="nHits", row_name="layerID", idx_name="cellID",
                 edep_name="edep", time_name="t", flat_name="vol_id",
                 signal_coding=1, geom=CyDet(), finalize_data=True):
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
        FlatHits.__init__(self, path=path, tree=tree, prefix=prefix,
                          branches=branches, hit_type_name=hit_type_name,
                          n_hits_name=n_hits_name, signal_coding=signal_coding,
                          finalize_data=False)

        # Get the geometry flat_IDs
        self.row_name = self.prefix + row_name
        self.idx_name = self.prefix + idx_name
        self.flat_name = self.prefix + flat_name

        # Get the geometry of the detector
        self._geom = geom

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
        self.data = np.rec.fromarrays(self.data, names=self.all_branches)
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
        for row, idx, hit in zip(row_data, idx_data, range(self.n_hits)):
            flat_ids[hit] = self._geom.point_lookup[row, idx]
        # Save this column and name it
        flat_id_column = flat_ids.astype(int)
        return flat_id_column

    def get_hit_vols(self, event_id, unique=True):
        """
        Returns the sequence of flat_ids that register hits in given event

        :return: numpy array of hit wires
        """
        # Select the relevant event from data
        flat_ids = self.get_events(event_id)[self.flat_name]
        if unique is True:
            flat_ids = np.unique(flat_ids)
        # TODO fix this error message
        #assert np.all(flat_ids >= 0), \
        #    'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0],
        #                                         self._geom.get)
        return flat_ids

    def get_sig_vols(self, event_id, unique=True):
        """
        Returns the sequence of flat_ids that register signal hits in given
        event

        :return: numpy array of hit wires
        """
        # Select the relevant event from data
        sig_vols = self.get_signal_hits(event_id)[self.flat_name]
        if unique is True:
            sig_vols = np.unique(sig_vols)
        return sig_vols

    def get_bkg_vols(self, event_id, unique=True):
        """
        Returns the sequence of flat_ids that register hits in given event

        :return: numpy array of hit wires
        """
        # Select the relevant event from data
        bkg_vols = self.get_background_hits(event_id)[self.flat_name]
        if unique is True:
            bkg_vols = np.unique(bkg_vols)
        return bkg_vols

    def get_hit_vector(self, event_id, unique=True):
        """
        Returns a vector denoting whether or not a wire has a hit on it. Returns
        1 for a hit, 0 for no hit

        :return: numpy array of shape [n_wires] whose value is 1 for a hit, 0
                 for no hit
        """
        # Get the flat vol IDs for those with hits
        hit_vols = self.get_hit_vols(event_id, unique=True)
        # Make the hit vector
        hit_vector = np.zeros(self._geom.n_points)
        hit_vector[hit_vols] = 1
        return hit_vector

    def get_hit_types(self, event_id, unique=True):
        """
        Returns hit type in all volumes, where signal is 1, background is 2,
        nothing is 0.  If signal and background are both incident, signal takes
        priority

        :return: numpy.array of shape [CyDet.n_points]
        """
        result = np.zeros(self._geom.n_points, dtype=int)
        # Get the background hits
        bkg_hits = np.unique(self.get_background_hits(event_id)[self.flat_name])
        result[bkg_hits] = 2
        # Get the signal hits
        sig_hits = np.unique(self.get_signal_hits(event_id)[self.flat_name])
        result[sig_hits] = 1
        return result.astype(int)

    def get_measurement(self, event_id, name):
        """
        Returns requested measurement in volumes, returning zero if the volume
        does not register this measurement

        :return: numpy.array of shape [CyDet.n_points]
        """
        result = np.zeros(self._geom.n_points, dtype=float)
        # Select the relevant event from data
        meas = self.get_events(event_id)[name]
        # Get the wire_ids of the hit data
        wire_ids = self.get_hit_vols(event_id, unique=False)
        # Add the measurement to the correct cells in the result
        result[wire_ids] += meas
        return result

    def get_energy_deposits(self, event_id):
        """
        Returns energy deposit in all wires

        :return: numpy.array of shape [CyDet.n_points]
        """
        energy_deposit = self.get_measurement(event_id, self.edep_name)
        return energy_deposit

    def get_hit_time(self, event_id):
        """
        Returns the timing of the hit

        :return: numpy.array of shape [CyDet.n_points]
        """
        time_hit = self.get_measurement(event_id, self.time_name)
        return time_hit


class CyDetHits(GeomHits):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, path="../data/signal.root", tree='tree', branches="mt",
                 prefix="CdcCell", hit_type_name="hittype", n_hits_name="nHits",
                 row_name="layerID", idx_name="cellID", flat_name="vol_id",
                 time_name="tstart", edep_name="edep", signal_coding=1,
                 finalize_data=True):
        """
        This generates hit data in a structured array from an input root file
        from a file. It assumes the naming convention "CdcCell_"+ variable for
        all leaves. It overlays its data on the uses the CyDet class to define
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
        GeomHits.__init__(self, path=path, tree=tree,
                          branches=branches, prefix="CdcCell",
                          hit_type_name=hit_type_name, n_hits_name=n_hits_name,
                          row_name=row_name, idx_name=idx_name,
                          time_name=time_name, edep_name=edep_name,
                          flat_name=flat_name, signal_coding=signal_coding,
                          geom=CyDet(), finalize_data=False)
        self.cydet = self._geom

        # Finialize the data if this is the final form
        if finalize_data:
            self._finalize_data()

    def get_hit_wires_even_odd(self, event_id):
        """
        Returns two sequences of wire_ids that register hits in given event, the
        first is only in even layers, the second is only in odd layers

        :return: numpy array of hit wires
        """
        hit_wires = self.get_hit_vols(event_id)
        odd_wires = np.where((self._geom.point_pol == 1))[0]
        even_hit_wires = np.setdiff1d(hit_wires, odd_wires, assume_unique=True)
        odd_hit_wires = np.intersect1d(hit_wires, odd_wires, assume_unique=True)
        return even_hit_wires, odd_hit_wires

    def get_hit_vector_even_odd(self, event_id):
        """
        Returns a vector denoting whether or not a wire on an odd layer has a
        hit on it. Returns 1 for a hit in an odd layer, 0 for no hit and all
        even layers

        :return: numpy array of shape [n_wires] whose value is 1 for a hit on an
                odd layer, 0 otherwise
        """
        even_wires, odd_wires = self.get_hit_wires_even_odd(event_id)
        even_hit_vector = np.zeros(self.cydet.n_points)
        even_hit_vector[even_wires] = 1
        odd_hit_vector = np.zeros(self.cydet.n_points)
        odd_hit_vector[odd_wires] = 1
        return even_hit_vector, odd_hit_vector

    def get_hit_time(self, event_id):
        """
        Returns the timing of the hit

        :return: numpy.array of shape [CyDet.n_points]
        """
        time_hit = self.get_measurement(event_id, self.prefix + "tstart")
        return time_hit

    def get_trigger_time(self, event_id):
        """
        Returns the timing of the trigger on an event

        :return: numpy.array of shape [CyDet.n_points]
        """
        # Check the trigger time has been set
        assert "CdcCell_mt" in self.all_branches,\
                "Trigger time has not been set yet"
        return self.get_measurement(event_id, self.prefix + "mt")

    def get_relative_time(self, event_id):
        """
        Returns the difference between the start time of the hit and the time of
        the trigger.  This value is capped to the time window of 1170 ns
        :return: numpy array of (t_start_hit - t_trig)%1170
        """
        trig_time = self.get_trigger_time(event_id)
        hit_time = self.get_hit_time(event_id)
        return hit_time - trig_time

### DEPRECIATED METHODS INCLUDED FOR BACKWARDS COMPATIBILITY ###

    def get_sig_wires(self, event_id):
        """
        Get all the signal wires in a given event.  This method is depreciated
        and simply wraps get_sig_vols
        """
        return self.get_sig_vols(event_id)

    def get_bkg_wires(self, event_id):
        """
        Get all the background wires in a given event.  This method is
        depreciated and simply wraps get_bkg_vols
        """
        return self.get_bkg_vols(event_id)

    def get_hit_wires(self, event_id):
        """
        Get all the hit wires in a given event.  This method is depreciated
        and simply wraps get_hit_vols
        """
        return self.get_hit_vols(event_id)

class SignalHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, cydet, version=1,
                 path="../data/signal.root", tree='tree',
                 signal_hits_only=False):
        """
        This generates hit data from a file in which both background and signal
        are included and coded. It assumes the naming convention
        "CdcCell_"+ variable for all leaves. It over lays its data on the uses
        the CyDet class to define its geometry.
        :param path: path to rootfile
        :param version: Version of SimChen data we are using
        :param tree: name of the tree in root dataset
        """

        self.data = root2array(path, treename=tree)
        self.cydet = cydet
        self.prefix = "CdcCell"
        self.n_events = len(self.data)
        self.version = version
        # TODO fix this
        self.signal_hits_only = signal_hits_only

    def get_hit_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event
        :return: numpy array of hit wires
        """
        # Select the relevant event from data
        event = self.data[event_id]
        # Recover an ordered list of wire_index and corresponding wire_ids
        wire_index = event[self.prefix + "_cellID"]
        layer_ids = event[self.prefix + "_layerID"]
        # Flatten these into the point_ids from the cydet
        wire_ids = self.cydet.point_lookup[layer_ids, wire_index]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0],
                                                 wire_index[wire_ids < 0])
        return wire_ids

    def get_hit_vector(self, event_id):
        """
        Returns a vector denoting whether or not a wire has a hit on it. Returns
        1 for a hit, 0 for no hit
        :return: numpy array of shape [n_wires] whose value is 1 for a hit, 0
                 for no hit
        """
        hit_vector = np.zeros(self.cydet.n_points)
        hit_vector[self.get_hit_wires(event_id)] = 1
        return hit_vector

    def get_hit_wires_even_odd(self, event_id):
        """
        Returns two sequences of wire_ids that register hits in given event, the
        first is only in even layers, the second is only in odd layers
        :return: numpy array of hit wires
        """
        hit_wires = self.get_hit_wires(event_id)
        odd_wires = np.where((self.cydet.point_pol == 1))[0]
        even_hit_wires = np.setdiff1d(hit_wires, odd_wires, assume_unique=True)
        odd_hit_wires = np.intersect1d(hit_wires, odd_wires, assume_unique=True)
        return even_hit_wires, odd_hit_wires

    def get_hit_vector_even_odd(self, event_id):
        """
        Returns a vector denoting whether or not a wire on an odd layer has a
        hit on it. Returns 1 for a hit in an odd layer, 0 for no hit and all
        even layers
        :return: numpy array of shape [n_wires] whose value is 1 for a hit on an
                odd layer, 0 otherwise
        """
        even_wires, odd_wires = self.get_hit_wires_even_odd(event_id)
        even_hit_vector = np.zeros(self.cydet.n_points)
        even_hit_vector[even_wires] = 1
        odd_hit_vector = np.zeros(self.cydet.n_points)
        odd_hit_vector[odd_wires] = 1
        return even_hit_vector, odd_hit_vector

    def get_measurement(self, event_id, name):
        """
        Returns requested measurement in all wires in requested event
        :return: numpy.array of shape [CyDet.n_points]
        """
        result = np.zeros(self.cydet.n_points, dtype=float)
        # Select the relevant event from data
        event = self.data[event_id]
        # Get the wire_ids of the hit data
        wire_ids = self.get_hit_wires(event_id)
        # Select the leaf of interest from the data
        measurement = event[name]
        # Add the measurement to the correct cells in the result
        result[wire_ids] += measurement
        return result

    def get_energy_deposits(self, event_id):
        """
        Returns energy deposit in all wires
        :return: numpy.array of shape [CyDet.n_points]
        """
        energy_deposit = self.get_measurement(event_id, self.prefix + "_edep")
        return energy_deposit

    def get_hit_time(self, event_id):
        """
        Returns the timing of the hit
        :return: numpy.array of shape [CyDet.n_points]
        """
        time_hit = self.get_measurement(event_id, self.prefix + "_tstart")
        return time_hit

    def get_trigger_time(self, event_id):
        """
        Returns energy deposit in all wires
        :return: numpy.array of shape [CyDet.n_points]
        """
        event = self.data[event_id]
        this_trig_time = event[self.prefix + "_mt"]
        trig_time = np.zeros((self.cydet.n_points))
        trig_time[self.get_hit_wires(event_id)] = this_trig_time
        return trig_time

    def get_relative_time(self, event_id):
        """
        Returns the difference between the start time of the hit and the time of
        the trigger.  This value is capped to the time window of 1170 ns
        :return: numpy array of (t_start_hit - t_trig)%1170
        """
        trig_time = self.get_trigger_time(event_id)
        hit_time = self.get_hit_time(event_id)
        # return np.remainder(hit_time - trig_time, 1170)
        return np.remainder(hit_time - trig_time, 1170)

    def get_time_neighbours_metric(self, event_id):
        """
        Returns a non-physical value which is largest for hits with no
        left-right neighbouring hits, large for hits who's neighbouring hits
        happen at a very different times, and small for cells whose neighbours
        happen at a very close time
        :return: numpy array non-physical measure of how close in time
                 LR neighbouring hits are
        """
        t_hits = self.get_hit_time(event_id)
        hit_wires = self.get_hit_wires(event_id)
        result = np.zeros(self.cydet.n_points)
        for wire in hit_wires:
            for shift in [1, -1]:
                sh_wire = self.cydet.shift_wire(wire, shift)
                t_metric = abs((t_hits[sh_wire] + 1) / (t_hits[wire] + 1))
                result[wire] += t_metric
        return result

    def get_hit_types(self, event_id):
        """
        Returns hit type in all wires, where signal is 1, background is 2,
        nothing is 0
        :return: numpy.array of shape [CyDet.n_points]
        """
        result = np.zeros(self.cydet.n_points, dtype=int)
        # Select the relevant event from data
        event = self.data[event_id]
        # Get the wire_ids of the hit data
        wire_ids = self.get_hit_wires(event_id)
        # TODO fix this
        if self.signal_hits_only:
            # Select the hit type leaf
            hit_types = event[self.prefix + "_tid"]
            coding = [1, 0]
        else:
            hit_types = event[self.prefix + "_hittype"]
            # Check which version we are using
            if self.version == 1:
                # Define custom coding
                pos_types = hit_types
                coding = [1, 2, 2, 2]
            elif self.version == 2:
                # Define custom coding
                pos_types = hit_types + 2
                coding = [2, 2, 2, 1, 1]
            elif self.version == 3:
                # Define custom coding
                pos_types = hit_types + 4
                coding = [2, 2, 2, 2, 2, 1, 2, 2]
        # Maps signal to 1, background to 2, and nothing to 0
        pos_types = np.take(coding, pos_types)
        # Add the measurement to the correct cells in the result
        result[wire_ids] += pos_types
        return result.astype(int)

    def get_sig_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register signal hits in
        given event
        :return: numpy array of signal hit wires
        """
        # Get all hit wires
        hit_types = self.get_hit_types(event_id)
        # Select signal hits
        sig_wires = np.where(hit_types == 1)[0]
        return sig_wires

    def get_bkg_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register background hits in
        given event
        :return: numpy array of signal hit wires
        """
        # Get all hit wires
        hit_types = self.get_hit_types(event_id)
        # Select signal hits
        bkg_wires = np.where(hit_types == 2)[0]
        return bkg_wires


class AllHits(SignalHits):
    def __init__(self, path="../data/signal_TDR.root", tree='tree', version=1,
                 signal_hits_only=False):
        cydet = CyDet()
        SignalHits.__init__(self, cydet, version, path, tree,
                            signal_hits_only=signal_hits_only)


class BackgroundHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, cydet, path="../data/proton_from_muon_capture",
                 tree='tree', hits=1000):
        """
        This generates hit data from a file in which both only background hits
        exist. It resamples the input file until to generate events with the
        desired number of hits.  It assumes the naming convention "O_"+ variable
        for all leaves. It over lays its data on the uses the CyDet class to
        define its geometry. Note that n_events here refers to the number of
        events used in resampling process.

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        """

        self.data = root2array(path, treename=tree)
        self.cydet = cydet
        self.prefix = "O"
        self.n_events = len(self.data)
        self.evt_random = Random()
        self.n_hits = hits
        initial_event = 0
        self.this_sample = 0
        self.event_index = 10
        self._get_sample(initial_event)

    def _get_sample(self, event_id):
        """
        Generates a scipy sparce matrix respresenting a full event, constructed
        from resampled events.The event_id is used as a random number seed for
        reproducibility.  The generated matrix maps from resampled event index
        and hit wire, to hit wire location in fully construced event

        Generates to internal field

         scipy.sparse.csr of shape [n_events, CyDet.n_points] which
         defines the events used and corresponding hit wires in them.
         The value of the matrix is the new hit wire in the
         reconstructed event, which is the original hit rotated by a
         random value
        """
        if self.event_index != event_id:
            # otherwise everything was done previously and cached
            #print "Getting sample {}".format(event_id)
            self.this_sample = lil_matrix((self.n_events, self.cydet.n_points),
                                          dtype=np.int16)
            # Seed the random number
            self.evt_random.seed(event_id)
            # Keep track of how many wires have been added from the samples
            n_wires = 0
            while n_wires < self.n_hits:
                # Select an event randomly
                this_event = self.evt_random.randint(0, self.n_events - 1)
                # Find the hit wires in the event
                wires = self.get_wires(this_event)
                # Add these to the total count
                n_wires += len(wires)
                # Rotate the wires a random amount around the layer
                rot = self.evt_random.random()
                new_wires = [self.cydet.rotate_wire(w, rot) for w in wires]
                # Add one to all new wire indecies to avoid problem with
                # explicit zeros in numpy.sparse matrix
                new_wires = [n_w + 1 for n_w in new_wires]
                # Mark event for use in sample
                self.this_sample[this_event, wires] = new_wires
            # Return a row sliceable array
            self.this_sample = self.this_sample.tocsr()
            self.event_index = event_id

    def get_hit_wires(self, event_id):
        """
        Returns the hit wires of the fully constructed event after rotation

        :return: numpy array of hit wires
        """
        self._get_sample(event_id)
        hit_wires = find(self.this_sample)[2]
        return np.unique(hit_wires)

    def _get_true_wires(self, event_id):
        """
        Returns the hit wires of the fully constructed event before rotation

        :return: numpy array of hit wires
        """
        self._get_sample(event_id)
        true_wires = find(self.this_sample)[1]
        return np.unique(true_wires)

    def _get_sample_events(self, event_id):
        """
        Returns the indecies of the resampled events used in the fully
        constructed event

        :return: numpy array of event_ids
        """
        self._get_sample(event_id)
        sample_events = find(self.this_sample)[0]
        return np.unique(sample_events)

    def _get_new_wire_ids(self, event_id, event_index):
        """
        Function to mask the use of shifting the new wire_ids when stored in the
        sparse matrix to avoid problem with explicit zero
        :param event_id: id of generated event
        :param event_index: index of sampled event in data file

        :return: rotated hit wire location in generated event, event_id, whose
                 data corresponds to the hit wires in event_index
        """
        self._get_sample(event_id)
        new_wires = find(self.this_sample[event_index, :])[2] - 1
        return new_wires

    def get_wires(self, event_index):
        """
        Returns the sequence of wire_ids that register hits in given sample
        event

        :return: numpy array of hit wires
        """
        # Select the relevant sampled event
        event = self.data[event_index]
        # Select the relevant sampled event
        wire_index = event[self.prefix + "_cellID"]
        layer_ids = event[self.prefix + "_layerID"]
        # Get the relevant wire_ids
        wire_ids = self.cydet.point_lookup[layer_ids, wire_index]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0],
                                                 wire_index[wire_ids < 0])
        return wire_ids

    def get_energy_deposits(self, event_id):
        """
        Returns the energy deposition in each wire by summing the contribution
        from each resampled event in the generated event

        :return: numpy.array of shape [CyDet.n_points]
        """
        energy_deposit = np.zeros(self.cydet.n_points)
        # Loop over the resampled events
        for event_index in self._get_sample_events(event_id):
            # Get the reampled event data
            event = self.data[event_index]
            # Get the wire_id's of the rotated wires using the sample map
            wire_ids = self._get_new_wire_ids(event_id, event_index)
            # Get the energy deposition of the true hit wires
            measurement = event[self.prefix + "_edep"]
            energy_deposit[wire_ids] += measurement
        return energy_deposit

    def get_hit_time(self, event_id):
        """
        Returns the energy deposition in each wire by taking the timing of the
        earliest hit from the resampled event that contibuted to the
        corresponding hit in the generated event

        :return: numpy.array of shape [CyDet.n_points]
        """
        time_hit = np.zeros(self.cydet.n_points)
        # Loop over the resampled events
        for event_index in self._get_sample_events(event_id):
            # Get the reampled event data
            event = self.data[event_index]
            # Get the wire_id's of the rotated wires using the sample map
            wire_ids = self._get_new_wire_ids(event_id, event_index)
            # Get the timing of the true hit wires
            timing = event[self.prefix + "_t"] % 1170
            # Noting that (timing) and (wire_ids) have corresponding order, open
            # a loop over the wires, noting the index in the wire_ids list
            # itself
            for place, wire in enumerate(wire_ids):
                # If this wire's entry has not been assigned, or if it is
                # greater that the new value, assign the new value
                if (time_hit[wire] == 0) | (time_hit[wire] > timing[place]):
                    time_hit[wire] = timing[place]
        return time_hit


class ResampledHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, sig_path="../data/signal.root", sig_tree='tree',
                 bkg_path_1="../data/proton_from_muon_capture_bg.root",
                 bkg_path_2="../data/muon_neutron_beam_bg.root",
                 bkg_path_3="../data/other_beam_bg.root",
                 bkg_tree='tree', occupancy=0.10):
        """
        This generates hit data from a file in which both background and signal
        are included and coded. It assumes the naming convention
        "CdcCell_"+ variable for all leaves. It over lays its data on the uses
        the CyDet class to define its geometry.

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        """

        self.cydet = CyDet()
        self.sig_hits = SignalHits(self.cydet, path=sig_path, tree=sig_tree)
        self.bkg_hits_1 = BackgroundHits(self.cydet, path=bkg_path_1, tree=bkg_tree)
        self.bkg_hits_2 = BackgroundHits(self.cydet, path=bkg_path_2, tree=bkg_tree)
        self.bkg_hits_3 = BackgroundHits(self.cydet, path=bkg_path_3, tree=bkg_tree)
        self.n_events = self.sig_hits.n_events
        self.event_index = 0
        total_bkg_hits = round(occupancy * self.cydet.n_points)
        total_avail_bkg = self.bkg_hits_1.n_events +\
                          self.bkg_hits_2.n_events +\
                          self.bkg_hits_3.n_events
        self.bkg_hits_1.n_hits = (total_bkg_hits) *\
                                 (self.bkg_hits_1.n_events/total_avail_bkg)
        self.bkg_hits_1.n_hits *= 0.15/0.43
        self.bkg_hits_1.n_hits = int(self.bkg_hits_1.n_hits)
        self.bkg_hits_2.n_hits = (total_bkg_hits)\
                                 * self.bkg_hits_2.n_events/total_avail_bkg
        self.bkg_hits_2.n_hits = int(self.bkg_hits_2.n_hits)
        self.bkg_hits_3.n_hits = (total_bkg_hits)\
                                 * self.bkg_hits_3.n_events/total_avail_bkg
        self.bkg_hits_3.n_hits = int(self.bkg_hits_3.n_hits)

    def get_hit_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event

        :return: numpy array of hit wires
        """
        sig_wires = self.sig_hits.get_hit_wires(event_id)
        bkg_1_wires = self.bkg_hits_1.get_hit_wires(event_id)
        bkg_2_wires = self.bkg_hits_2.get_hit_wires(event_id)
        bkg_3_wires = self.bkg_hits_3.get_hit_wires(event_id)
        wire_ids = np.append(sig_wires, bkg_1_wires)
        wire_ids = np.append(wire_ids, bkg_2_wires)
        wire_ids = np.append(wire_ids, bkg_3_wires)
        return np.unique(wire_ids)

    def get_energy_deposits(self, event_id):
        """
        Returns energy deposit in all wires

        :return: numpy.array of shape [CyDet.n_points]
        """
        sig_energy = self.sig_hits.get_energy_deposits(event_id)
        bkg_energy_1 = self.bkg_hits_1.get_energy_deposits(event_id)
        bkg_energy_2 = self.bkg_hits_2.get_energy_deposits(event_id)
        bkg_energy_3 = self.bkg_hits_3.get_energy_deposits(event_id)
        energy = sig_energy + bkg_energy_1  + bkg_energy_2 + bkg_energy_3
        return energy

    def get_sig_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register signal hits in
        given event

        :return: numpy array of signal hit wires
        """
        sig_wires = self.sig_hits.get_sig_wires(event_id)
        return sig_wires

    def get_bkg_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register background hits in
        given event

        :return: numpy array of signal hit wires
        """
        # Signal sample actually also has BG hits in it already
        bkg_wires = np.append(self.bkg_hits_1.get_hit_wires(event_id),
                              self.bkg_hits_2.get_hit_wires(event_id))
        bkg_wires = np.append(bkg_wires, self.bkg_hits_3.get_hit_wires(event_id))
        bkg_wires = np.append(bkg_wires, self.sig_hits.get_bkg_wires(event_id))
        bkg_wires = set(bkg_wires) - set(self.get_sig_wires(event_id))
        return np.array(list(bkg_wires))

    def get_hit_types(self, event_id):
        """
        Returns hit type in all wires, where signal is 1, background is 2,
        nothing is 0.  In the case of hit overlap between background an signal,
        signal is given priority. Note: The signal sample often has a few
        background hits already, mostly along the track

        :return: numpy.array of shape [CyDet.n_points]
        """
        result = np.zeros(self.cydet.n_points, dtype=int)
        result[self.get_bkg_wires(event_id)] = 2
        result[self.get_sig_wires(event_id)] = 1
        return result.astype(int)

class OnlyBackgroundHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, n_events=1000,
                 bkg_path_1="../data/proton_from_muon_capture_bg.root",
                 bkg_path_2="../data/muon_neutron_beam_bg.root",
                 bkg_path_3="../data/other_beam_bg.root",
                 bkg_tree='tree', occupancy=0.10):
        """
        This generates hit data from a file in which both background and signal
        are included and coded. It assumes the naming convention
        "CdcCell_"+ variable for all leaves. It over lays its data on the uses
        the CyDet class to define its geometry.

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        """

        self.cydet = CyDet()
        self.bkg_hits_1 = BackgroundHits(self.cydet, path=bkg_path_1, tree=bkg_tree)
        self.bkg_hits_2 = BackgroundHits(self.cydet, path=bkg_path_2, tree=bkg_tree)
        self.bkg_hits_3 = BackgroundHits(self.cydet, path=bkg_path_3, tree=bkg_tree)
        self.n_events = n_events
        self.event_index = 0
        total_bkg_hits = round(occupancy * self.cydet.n_points)
        total_avail_bkg = self.bkg_hits_1.n_events +\
                          self.bkg_hits_2.n_events +\
                          self.bkg_hits_3.n_events
        self.bkg_hits_1.n_hits = (total_bkg_hits) *\
                                float(self.bkg_hits_1.n_events)/total_avail_bkg
        self.bkg_hits_1.n_hits *= 0.15/0.43
        self.bkg_hits_1.n_hits = int(self.bkg_hits_1.n_hits)
        self.bkg_hits_2.n_hits = (total_bkg_hits)\
                                 * self.bkg_hits_2.n_events/total_avail_bkg
        self.bkg_hits_2.n_hits = int(self.bkg_hits_2.n_hits)
        self.bkg_hits_3.n_hits = (total_bkg_hits)\
                                 * self.bkg_hits_3.n_events/total_avail_bkg
        self.bkg_hits_3.n_hits = int(self.bkg_hits_3.n_hits)

    def get_hit_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event

        :return: numpy array of hit wires
        """
        bkg_1_wires = self.bkg_hits_1.get_hit_wires(event_id)
        bkg_2_wires = self.bkg_hits_2.get_hit_wires(event_id)
        bkg_3_wires = self.bkg_hits_3.get_hit_wires(event_id)
        wire_ids = np.append(bkg_1_wires, bkg_2_wires)
        wire_ids = np.append(wire_ids, bkg_3_wires)
        return np.unique(wire_ids)

    def get_energy_deposits(self, event_id):
        """
        Returns energy deposit in all wires

        :return: numpy.array of shape [CyDet.n_points]
        """
        bkg_energy_1 = self.bkg_hits_1.get_energy_deposits(event_id)
        bkg_energy_2 = self.bkg_hits_2.get_energy_deposits(event_id)
        bkg_energy_3 = self.bkg_hits_3.get_energy_deposits(event_id)
        energy = bkg_energy_1  + bkg_energy_2 + bkg_energy_3
        return energy

    def get_sig_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register signal hits in
        given event

        :return: numpy array of signal hit wires
        """
        return []

    def get_bkg_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register background hits in
        given event

        :return: numpy array of signal hit wires
        """
        # Signal sample actually also has BG hits in it already
        bkg_wires = np.append(self.bkg_hits_1.get_hit_wires(event_id),
                              self.bkg_hits_2.get_hit_wires(event_id))
        bkg_wires = np.append(bkg_wires, self.bkg_hits_3.get_hit_wires(event_id))
        bkg_wires = set(bkg_wires) - set(self.get_sig_wires(event_id))
        return np.array(list(bkg_wires))

    def get_hit_types(self, event_id):
        """
        Returns hit type in all wires, where signal is 1, background is 2,
        nothing is 0.  In the case of hit overlap between background an signal,
        signal is given priority. Note: The signal sample often has a few
        background hits already, mostly along the track

        :return: numpy.array of shape [CyDet.n_points]
        """
        result = np.zeros(self.cydet.n_points, dtype=int)
        result[self.get_bkg_wires(event_id)] = 2
        return result.astype(int)

class SummedHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, sig_path="../data/trimmed_signal.root", sig_tree='tree',
                 sig_version=1,
                 bkg_path="../data/151208_SimChen_noise.root", bkg_tree='tree',
                 bkg_version=2,
                 time_seed=0):
        """
        This generates hit data from a file two files, one with background only,
        and one with signal only.  Currently, it assumes all trigger signals are
        from the signal file.  In the instance that signal and background hit
        the same wire, the hit is considered signal, the energy depositions are
        summed, and the earlier hit time is taken

        :param time_seed:  Seed for shifting the signal hits and trigger timing
                           by a random amount
        """

        self.cydet = CyDet()
        self.sig_hits = SignalHits(self.cydet, path=sig_path, tree=sig_tree,
                                   version=sig_version)
        self.bkg_hits = SignalHits(self.cydet, path=bkg_path, tree=bkg_tree,
                                   version=bkg_version)
        self.n_events = self.bkg_hits.n_events
        self.time_seed = time_seed

    def get_hit_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event

        :return: numpy array of hit wires
        """
        sig_wires = self.sig_hits.get_hit_wires(event_id)
        bkg_wires = self.bkg_hits.get_hit_wires(event_id)
        wire_ids = np.append(sig_wires, bkg_wires)
        return np.unique(wire_ids)

    def get_energy_deposits(self, event_id):
        """
        Returns energy deposit in all wires

        :return: numpy.array of shape [CyDet.n_points]
        """
        sig_energy = self.sig_hits.get_energy_deposits(event_id)
        bkg_energy = self.bkg_hits.get_energy_deposits(event_id)
        energy = sig_energy + bkg_energy
        return energy

    def get_sig_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register signal hits in
        given event

        :return: numpy array of signal hit wires
        """
        sig_wires = self.sig_hits.get_sig_wires(event_id)
        return sig_wires

    def get_bkg_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register background hits in
        given event

        :return: numpy array of signal hit wires
        """
        # Signal sample actually also has BG hits in it already
        bkg_wires = self.bkg_hits.get_bkg_wires(event_id)
        sig_wires = self.bkg_hits.get_bkg_wires(event_id)
        return np.unique(np.append(bkg_wires, sig_wires))

    def get_hit_types(self, event_id):
        """
        Returns hit type in all wires, where signal is 1, background is 2,
        nothing is 0.  In the case of hit overlap between background an signal,
        signal is given priority. Note: The signal sample often has a few
        background hits already, mostly along the track

        :return: numpy.array of shape [CyDet.n_points]
        """
        result = np.zeros(self.cydet.n_points, dtype=int)
        result[self.get_bkg_wires(event_id)] = 2
        result[self.get_sig_wires(event_id)] = 1
        return result.astype(int)

    def get_hit_vector(self, event_id):
        """
        Returns a vector denoting whether or not a wire has a hit on it. Returns
        1 for a hit, 0 for no hit

        :return: numpy array of shape [n_wires] whose value is 1 for a hit, 0
                 for no hit
        """
        hit_vector = np.zeros(self.cydet.n_points)
        hit_vector[self.get_hit_wires(event_id)] = 1
        return hit_vector

    def get_hit_wires_even_odd(self, event_id):
        """
        Returns two sequences of wire_ids that register hits in given event, the
        first is only in even layers, the second is only in odd layers

        :return: numpy array of hit wires
        """
        hit_wires = self.get_hit_wires(event_id)
        odd_wires = np.where((self.cydet.point_pol == 1))[0]
        even_hit_wires = np.setdiff1d(hit_wires, odd_wires, assume_unique=True)
        odd_hit_wires = np.intersect1d(hit_wires, odd_wires, assume_unique=True)
        return even_hit_wires, odd_hit_wires

    def get_hit_vector_even_odd(self, event_id):
        """
        Returns a vector denoting whether or not a wire on an odd layer has a
        hit on it. Returns 1 for a hit in an odd layer, 0 for no hit and all
        even layers

        :return: numpy array of shape [n_wires] whose value is 1 for a hit on an
                odd layer, 0 otherwise
        """
        even_wires, odd_wires = self.get_hit_wires_even_odd(event_id)
        even_hit_vector = np.zeros(self.cydet.n_points)
        even_hit_vector[even_wires] = 1
        odd_hit_vector = np.zeros(self.cydet.n_points)
        odd_hit_vector[odd_wires] = 1
        return even_hit_vector, odd_hit_vector

    def get_measurement(self, event_id, name, source='bkg'):
        """
        Returns requested measurement in all wires in requested event

        :param source: Which data file to pull the measurement from.
                       Allowed values are bkg and sig

        :return: numpy.array of shape [CyDet.n_points]
        """
        # Select the relevant data and event
        assert (source == 'bkg') or (source == 'sig'),\
               "Source must be \'sig\' or \'bkg\'"
        if source == 'bkg':
            return self.bkg_hits.get_measurement(event_id, name)
        elif source == 'sig':
            return self.sig_hits.get_measurement(event_id, name)

    def get_hit_time(self, event_id):
        """
        Returns the timing of the hit

        :return: numpy.array of shape [CyDet.n_points]
        """
        sig_time_hit = self.sig_hits.get_hit_time(event_id)
        bkg_time_hit = self.bkg_hits.get_hit_time(event_id)
        return np.minimum(sig_time_hit, bkg_time_hit)

    def get_trigger_time(self, event_id):
        """
        Returns energy deposit in all wires

        :return: numpy.array of shape [CyDet.n_points]
        """
        if self.sig_hits.version == 1:
            trig_time = self.sig_hits.get_trigger_time(event_id)
        else:
            # TODO support trigger timing for version 2, i.e. build own trigger
            # signal
            print "Error, signal version 2 not supported yet"
        return trig_time

    def get_relative_time(self, event_id):
        """
        Returns the difference between the start time of the hit and the time of
        the trigger.  This value is capped to the time window of 1170 ns

        :return: numpy array of (t_start_hit - t_trig)%1170
        """
        trig_time = self.get_trigger_time(event_id)
        hit_time = self.get_hit_time(event_id)
        # return np.remainder(hit_time - trig_time, 1170)
        return np.remainder(hit_time - trig_time, 1170)


