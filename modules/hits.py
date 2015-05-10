import numpy as np
from root_numpy import root2array
#import math
from cylinder import CyDet
from random import Random
from scipy.sparse import lil_matrix, find

"""
Notation used below:
 - wire_id is flat enumerator of all wires
 - layer_id is the index of layer
 - wire_index is the index of wire in the layer
"""

class AllHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, path="data/signal_TDR.root", treename='tree'):
        """
        This generates hit data from a file in which both background and signal
        are included and coded. It assumes the naming convention
        "CdcCell_"+ variable for all leaves. It over lays its data on the uses
        the CyDet class to define its geometry.

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """

        self.data = root2array(path, treename=treename)
        self.cydet = CyDet()
        self.prefix = "CdcCell"
        self.n_events = len(self.data)

    def get_hit_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event

        :return: numpy array of hit wires
        """
        # Select the relevant event from data
        event = self.data[event_id]
        # Recover an ordered list of wire_index and corresponding wire_ids
        wire_index = event[self.prefix+"_cellID"]
        layer_ids = event[self.prefix+"_layerID"]
        # Flatten these into the point_ids from the cydet
        wire_ids = self.cydet.point_lookup[layer_ids, wire_index]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0],
                                                 wire_index[wire_ids < 0])
        return wire_ids

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
        energy_deposit = self.get_measurement(event_id, self.prefix+"_edep")
        return energy_deposit

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
        # Select the hit type leaf
        measurement = event[self.prefix+"_hittype"]
        # Define custom coding
        coding = [1, 2, 2, 2]
        # Maps signal to 1, background to 2, and nothing to 0
        measurement = np.take(coding, measurement)
        # Add the measurement to the correct cells in the result
        result[wire_ids] += measurement
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

class BackgroundHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, bkg_path="data/proton_from_muon_capture",
                       bkg_tree='tree', hits=1000):
        """
        This generates hit data from a file in which both only background hits
        exist. It resamples the input file until to generate events with the
        desired number of hits.  It assumes the naming convention "O_"+ variable
        for all leaves. It over lays its data on the uses the CyDet class to
        define its geometry. Note that n_events here refers to the number of
        events used in resampling process.

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """

        self.bkg_data = root2array(bkg_path, treename=bkg_tree)
        self.cydet = CyDet()
        self.prefix = "O"
        self.n_events = len(self.bkg_data)
        self.evt_random = Random()
        self.n_hits = hits
        self.hits_by_event = self._prepare_events()
        self.this_sample = self.get_sample(0)

    def _prepare_events(self):
        """
        Constructs a sparse matrix of shape [n_events, CyDet.n_points] to record
        the hit wires of each event.

        :return: scipy.sparse.csr_matrix of shape [n_events, CyDet.n_points]
                 where non-zero entries correspond to the hit_wires of a given
                 sample event
        """
        events = lil_matrix((self.n_events, self.cydet.n_points))
        for evt in range(self.n_events):
            hit_wires = self.get_wires(evt)
            events[evt, hit_wires] = 1
        events = events.tocsr()
        return events

    def get_sample(self, event_id):
        """
        Returns a scipy sparce matrix respresenting a full event, constructed
        from resampled events.  The event_id is used as a random number seed for
        reproducibility.

        :return: scipy.sparse.csr of shape [n_events, CyDet.n_points] which
                 defines the events used and corresponding hit wires in them
        """
        #:param n_hits: option to override the desired number of hits defined at
        #               initiation
        self.this_sample = lil_matrix((self.n_events, self.cydet.n_points))
        # Seed the random number
        self.evt_random.seed(event_id)
        # Keep track of how many wires have been added from the samples
        n_wires = 0
        while n_wires < self.n_hits:
            # Select an event randomly
            this_event = self.evt_random.randint(0, self.n_events-1)
            # Find the hit wires in the event
            wires = find(self.hits_by_event[this_event, :])[1]
            # Add these to the total count
            n_wires += len(wires)
            # Rotate the wires a random amount around the layer
            rot = self.evt_random.random()
            new_wires = [self.cydet.rotate_wire(w, rot) for w in wires]
            # Mark event for use in sample
            self.this_sample[this_event, wires] = new_wires
        #Return a row sliceable array
        self.this_sample = self.this_sample.tocsr()
        return self.this_sample

    def get_hit_wires(self):
        """
        Returns the hit wires of the fully constructed event after rotation

        :return: numpy array of hit wires
        """
        hit_wires = find(self.this_sample)[2]
        return np.unique(hit_wires)

    def get_true_wires(self):
        """
        Returns the hit wires of the fully constructed event before rotation

        :return: numpy array of hit wires
        """
        true_wires = find(self.this_sample)[1]
        return np.unique(true_wires)

    def get_sample_events(self):
        """
        Returns the indecies of the resampled events used in the fully
        constructed event

        :return: numpy array of event_ids
        """
        sample_events = find(self.this_sample)[0]
        return np.unique(sample_events)

    def get_wires(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given sample
        event

        :return: numpy array of hit wires
        """
        # Select the relevant sampled event
        event = self.bkg_data[event_id]
        # Select the relevant sampled event
        wire_index = event[self.prefix+"_cellID"]
        layer_ids = event[self.prefix+"_layerID"]
        # Get the relevant wire_ids
        wire_ids = self.cydet.point_lookup[layer_ids, wire_index]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0],
                                                 wire_index[wire_ids < 0])
        return wire_ids

    def get_energy_deposits(self):
        """
        Returns the energy deposition in each wire by summing the contribution
        from each resampled event in the generated event

        :return: numpy.array of shape [CyDet.n_points]
        """
        energy_deposit = np.zeros(self.cydet.n_points)
        # Loop over the resampled events
        for event_id in self.get_sample_events():
            # Get the reampled event data
            event = self.bkg_data[event_id]
            # Get the wires from this resampled event
            wire_ids = self.get_hit_wires()
            # Get the energy deposition of the hit wires
            measurement = event[self.prefix+"_edep"]
            energy_deposit[wire_ids] += measurement
        return energy_deposit
