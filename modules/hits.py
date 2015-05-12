import numpy as np
from root_numpy import root2array
# import math
from cylinder import CyDet
from random import Random
from scipy.sparse import lil_matrix, find

"""
Notation used below:
 - wire_id is flat enumerator of all wires
 - layer_id is the index of layer
 - wire_index is the index of wire in the layer
"""


class SignalHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, cydet, path="../data/signal.root", tree='tree'):
        """
        This generates hit data from a file in which both background and signal
        are included and coded. It assumes the naming convention
        "CdcCell_"+ variable for all leaves. It over lays its data on the uses
        the CyDet class to define its geometry.

        :param path: path to rootfile
        :param tree: name of the tree in root dataset
        """

        self.data = root2array(path, treename=tree)
        self.cydet = cydet
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
        wire_index = event[self.prefix + "_cellID"]
        layer_ids = event[self.prefix + "_layerID"]
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
        energy_deposit = self.get_measurement(event_id, self.prefix + "_edep")
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
        measurement = event[self.prefix + "_hittype"]
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


class AllHits(SignalHits):
    def __init__(self, path="../data/signal_TDR.root", tree='tree'):
        cydet = CyDet()
        SignalHits.__init__(cydet, path, tree)


class BackgroundHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, cydet, path="data/proton_from_muon_capture",
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
            print "Getting sample {}".format(event_id)
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


class ResampledHits(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=relative-import
    def __init__(self, sig_path="../data/signal_TDR.root", sig_tree='tree',
                 bkg_path="../data/proton_from_muon_capture_bg.root", bkg_tree='tree',
                 occupancy=0.10):
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
        self.bkg_hits = BackgroundHits(self.cydet, path=bkg_path, tree=bkg_tree)
        self.n_events = self.sig_hits.n_events
        self.event_index = 0

        total_bkg_hits = round(occupancy * self.cydet.n_points)
        self.bkg_hits.n_hits = total_bkg_hits

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
        bkg_wires = np.append(self.bkg_hits.get_hit_wires(event_id),
                              self.sig_hits.get_bkg_wires(event_id))
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
        result[self.get_bkg_wires] = 2
        result[self.get_sig_wires] = 1
        return result.astype(int)
