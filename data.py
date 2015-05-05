import numpy as np
from scipy.stats import norm
from root_numpy import root2array
import math
from scipy.sparse import *

"""
Notation used below:
 - wire_id is flat enumerator of all wires (from 0 to 4985)
 - layer_id is the index of layer (from 0 to 19)
 - cell_id is the index of wire in the layer (from 0 to layer_size -1)
"""


class Dataset:
    def __init__(self, path="data/signal_TDR.root", treename='tree'):
        """
        Dataset provides an interface to work with MC stored in root format.
        Results of methods are either numpy.arrays or scipy.sparse objects.
        Note that root data enumerates layer id's from [0-17].  These correspond
        to [1-18] in this structure

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """
        self.hits_data = root2array(path, treename=treename)
        # Hardcode information about wires in the CDC
        self.n_wires_in_layers = [198, 198, 204, 210, 216, 222, 228, 234, 240, 246,
                                  252, 258, 264, 270, 276, 282, 288, 294, 300, 306]
        self.first_wire = self._get_first_wire()
        self.total_wires = 4986
        assert sum(self.n_wires_in_layers) == self.total_wires
        self.r_layers = [51.4, 53, 54.6, 56.2, 57.8, 59.4, 61, 62.6, 64.2, 65.8,
                         67.4, 69, 70.6, 72.2, 73.8, 75.4, 77, 78.6, 80.2, 81.8]
#        self.start_phi_layer = [0.00000, 0.015867, 0.015400, 0.000000, 0.014544,
#                                0.00000, 0.000000, 0.013426, 0.000000, 0.012771,
#                                0.00000, 0.012177, 0.000000, 0.011636, 0.000000,
#                                0.00000, 0.000000, 0.010686, 0.000000, 0.010267]
        self.start_phi_layer = [0.00000, 0.015867, 0.000000, 0.000000, 0.000000,
                                0.00000, 0.014960, 0.014960, 0.000000, 0.000000,
                                0.00000, 0.000000, 0.000000, 0.000000, 0.000000,
                                0.00000, 0.000000, 0.000000, 0.000000, 0.000000]
        self.signal_r = 30. # determined from truth distribution of radial
                            # coordinates of hits
        self.signal_r_sigma = 2 # defines the spread of the smearing of the
                                # signal track from the constant value
        self.target_r = 20. # defined to cover the entire sense volume
        self.track_smear = 5   # defines the number of cells the track
                               # correspondence function will use when
                               # calculating probabilites
        self.n_track_phi_bins = 200
        self.d_arc_track_bins = 0.5
        self.n_track_rho_bins = 50
        self.n_track_bins = self.n_track_phi_bins*self.n_track_rho_bins

        self.d_phi_layer = self._calculated_d_phi()
        self.lookup_table = self._prepare_wires_lookup()
        self.radii_table = self._prepare_wire_radius_lookup()
        self.angles_table = self._prepare_wire_angles_lookup()
        self.neighbours_table = self.get_neighbours()

        self.track_table = self._prepare_track_lookup()
        self.track_phis = self._prepare_track_phis()
        self.track_rhos = self._prepare_track_rhos()
        self.correspondence = self._prepare_wire_track_correspondence()

    @property
    def n_events(self):
        return len(self.hits_data)

    def _get_first_wire(self):
        first_wire = np.zeros(20, dtype = int)
        for i in range(len(self.n_wires_in_layers)):
            first_wire[i] = sum(self.n_wires_in_layers[:i])
        return first_wire

    def _prepare_wires_lookup(self):
        """
        Prepares lookup table to map from [layer_id, cell_id] -> wire_id
        First 198 and last 306 wires not used currently
        :return:
        """
        lookup = np.zeros([len(self.n_wires_in_layers), max(self.n_wires_in_layers)], dtype='int')
        lookup[:, :] = - 1
        wire_id = 0
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            for cell_id in range(layer_size):
                lookup[layer_id, cell_id] = wire_id
                wire_id += 1
        assert wire_id == sum(self.n_wires_in_layers)
        return lookup

    def _prepare_wire_radius_lookup(self):
        """
        Prepares lookup table to map from wire id to the radial position
        :return: numpy.array of shape [total_wires]
        """
        first_wire = 0
        radii = np.zeros(self.total_wires, dtype=float)
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            radii[first_wire:first_wire + layer_size] = self.r_layers[layer_id]
            first_wire += layer_size
        return radii

    def _prepare_wire_angles_lookup(self):
        """
        Prepares lookup table to map from wire id to the angular position
        :return: numpy.array of shape [total_wires]
        """
        angles = np.zeros(self.total_wires, dtype=float)
        first_wire = 0
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            for wire_index in range(layer_size):
                angles[first_wire + wire_index] = (self.start_phi_layer[layer_id]
                    + self.d_phi_layer[layer_id] * wire_index)
            first_wire += layer_size
        angles %= 2*math.pi
        return angles

    def _get_wire_ids(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event
        """
        event = self.hits_data[event_id]
        cell_ids = event["CdcCell_cellID"]
        # + 1 since first is insensetive, i.e. layer 0 in root file is layer 1
        # in this structure
        layer_ids = event["CdcCell_layerID"] + 1
        wire_ids = self.lookup_table[layer_ids, cell_ids]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0], cell_ids[wire_ids < 0])
        return wire_ids

    def _calculated_d_phi(self):
        """
        Returns the phi separation of the wires as defined by the number of
        wires in the layer
        """
        return 2*math.pi/np.asarray(self.n_wires_in_layers)

    def get_measurement(self, event_id, name):
        """
        Returns requested measurement in all wires in requested event
        :return: numpy.array of shape [total_wires]
        """
        event = self.hits_data[event_id]
        wire_ids = self._get_wire_ids(event_id)
        measurement = event[name]
        result = np.zeros(self.total_wires, dtype=float)
        result[wire_ids] += measurement
        return result

    def get_energy_deposits(self, event_id):
        """
        Returns energy deposit in all wires
        :return: numpy.array of shape [total_wires]
        """
        energy_deposit = self.get_measurement(event_id, "CdcCell_edep")
        return energy_deposit

    def get_hit_types(self, event_id):
        """
        Returns hit type in all wires, where signal is 1, background is 2,
        nothing is 0
        :return: numpy.array of shape [total_wires]
        """
        event = self.hits_data[event_id]
        wire_ids = self._get_wire_ids(event_id)
        measurement = event["CdcCell_hittype"]
        coding = [1, 2, 2, 2]
        # Maps signal to 1, background to 2, and nothing to 0
        measurement = np.take(coding, measurement)
        result = np.zeros(self.total_wires, dtype=int)
        result[wire_ids] += measurement
        return result.astype(int)

    def get_neighbours(self):
        """
        Returns a sparse array of neighbour relations
        :return: scipy.sparse of shape [total_wires,total_wires]
        """
        neighbours = lil_matrix((self.total_wires, self.total_wires))
        first_wire = 0
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            for wire_index in range(layer_size):
                this_wire = wire_index + first_wire
                next_wire = first_wire + (wire_index + 1)%layer_size
                neighbours[next_wire, this_wire] = 1  # Clockwise
                neighbours[this_wire, next_wire] = 1  # Anti-Clockwise
                if (layer_id != len(self.n_wires_in_layers) - 1) :
                    wire_a = self.angles_table[this_wire]
                    angle_win = self.d_phi_layer[layer_id + 1] * 1.5
                    above = np.where(
                         (self.radii_table == self.r_layers[layer_id + 1]) &
                         ((abs(wire_a - self.angles_table) < angle_win) |
                         (2*math.pi - abs(wire_a - self.angles_table) < angle_win)
                         ))[0]
                    neighbours[this_wire, above[:]] = 1 # Above
                if (layer_id != 0):
                    wire_a = self.angles_table[this_wire]
                    angle_win = self.d_phi_layer[layer_id - 1] * 1.5
                    below = np.where(
                         (self.radii_table == self.r_layers[layer_id - 1]) &
                         ((abs(wire_a - self.angles_table) < angle_win) |
                         (2*math.pi - abs(wire_a - self.angles_table) < angle_win)
                         ))[0]
                    neighbours[this_wire, below[:]] = 1 # Below
            first_wire += layer_size
        return neighbours

    def get_wires_rhos_and_phis(self):
        """
        Returns the positions of each wire in radial system

        :return: pair of numpy.arrays of shape [n_wires],
         - first one contains rho`s (radii)
         - second one contains phi's (angles)
        """
        return self.radii_table, self.angles_table

    def get_wires_xs_and_ys(self):
        """
        Returns the positions of each wire in cartesian system

        :return: pair of numpy.arrays of shape [n_wires],
         - first one contains x`s
         - second one contains y's
        """
        x = self.radii_table * np.cos(self.angles_table)
        y = self.radii_table * np.sin(self.angles_table)
        return x, y

    def _prepare_track_lookup(self):
        """
        Prepares lookup table to map from [rho_bin, phi_bin] -> bin_id
        :return: numpy.array of shape [n_track_bins]
        """
        track_lookup = np.zeros([self.n_track_rho_bins,
                                 self.n_track_phi_bins], dtype='int')
        track_lookup[:, :] = - 1
        track_bin = 0
        for rho_bin in range(self.n_track_rho_bins):
            for phi_bin in range(self.n_track_phi_bins):
                track_lookup[rho_bin, phi_bin] = track_bin
                track_bin += 1
        assert track_bin == self.n_track_rho_bins*self.n_track_phi_bins
        return track_lookup

    def _prepare_track_rhos(self):
        """
        Returns the physical locations of each track_bin in rho.  
        
        Maximal distance is defined as the location where the signal track will
        enter the last layer.  

        Minimal distance defined as the distance where the track+track_smear  
        will enter the first layer, provided this distance is not less than the
        physics allows (i.e. track must pass through both target and detector
        region)

        :return: numpy.array of shape [n_track_bins]
        """
        t_0 = 0
        track_rhos = np.zeros(self.n_track_bins)
        r_max = self.r_layers[-2] - self.signal_r
        r_min = max(self.signal_r - self.target_r, 
                    self.r_layers[1] - self.signal_r - self.track_smear)
        drho = (r_max - r_min)/(self.n_track_rho_bins-1)
        for n in range(self.n_track_rho_bins):
            track_rhos[t_0:t_0 + self.n_track_phi_bins] = r_min + drho*n 
            t_0 += self.n_track_phi_bins
        return track_rhos

     def _prepare_track_phis(self):
        """
        Discretizes the possible locations of the center of a track in phi

        :return: numpy.array of shape [n_track_bins], contains possible
        centers of the tracks in phi
        """
        dphi = (2*math.pi)/self.n_track_phi_bins
        return np.fromfunction(lambda x:(x%self.n_track_phi_bins)*dphi,
                              (self.n_track_bins,))

    def get_tracks_rhos_and_phis(self):
        """
        Returns the positions of each track center 

        :return: pair of numpy.arrays of shape [n_track_bins],
         - first one contains rho`s (radii)
         - second one contains phi's (angles)
        """
        return self.track_rhos, self.track_phis

    def polar_dist(self,rho_1,phi_1,rho_2,phi_2):
        """
        Returns Euclidian distance between to points in polar coordinates
        """
        return np.sqrt(rho_1**2 + rho_2**2 - 2*rho_1*rho_2*np.cos(phi_1-phi_2))

    def dist_prob(self, distance):
        """
        Returns probability that hit on wire belongs to given track bin
        """
        return norm.pdf(distance, scale=self.signal_r_sigma)

    def trk_wire_dist(self, wr_id, t_bin):
        """
        Returns distance between a wire given by wr_id and the center of a
        potential track given by t_bin
        """
        return self.polar_dist(self.radii_table[wr_id],self.angles_table[wr_id],
                               self.track_rhos[t_bin], self.track_phis[t_bin])

    def trk_wire_prob(self, wr_id, t_bin):
        """
        Returns probability that hit on wire belongs to given track bin
        """
        return norm.pdf(self.trk_wire_dist(wr_id, t_bin) - self.signal_r,
                        scale=self.signal_r_sigma)

    def _prepare_wire_track_correspondence(self):
        """
        Defines the probability that a given wire belongs to a track centered at
        a given track center bin
        :returns: scipy.sparse matrix of shape [n_wires, n_track_bin]
        """
        corresp = lil_matrix((self.total_wires,self.n_track_bins))
        for trk_bin in range(self.n_track_bins):
            for wire_id in range(self.total_wires):
                this_dist = self.trk_wire_dist(wire_id,trk_bin) - self.signal_r
                if abs(this_dist) < self.track_smear:
                    corresp[wire_id,trk_bin] = self.dist_prob(this_dist)
        return corresp




class Dataset_SimChen:
    def __init__(self, path="data/signal_TDR.root", treename='tree'):
        """
        Dataset provides an interface to work with MC stored in root format.
        Results of methods are either numpy.arrays or scipy.sparse objects.
        Note that root data enumerates layer id's from [0-17].  These correspond
        to [1-18] in this structure

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """
        self.hits_data = root2array(path, treename=treename)
        # Hardcode information about wires in the CDC
        self.n_wires_in_layers = [198, 198, 204, 210, 216, 222, 228, 234, 240, 246,
                                  252, 258, 264, 270, 276, 282, 288, 294, 300, 306]
        self.first_wire = self._get_first_wire()
        self.total_wires = 4986
        assert sum(self.n_wires_in_layers) == self.total_wires
        self.r_layers = [51.4, 53, 54.6, 56.2, 57.8, 59.4, 61, 62.6, 64.2, 65.8,
                         67.4, 69, 70.6, 72.2, 73.8, 75.4, 77, 78.6, 80.2, 81.8]
#        self.start_phi_layer = [0.00000, 0.015867, 0.015400, 0.000000, 0.014544,
#                                0.00000, 0.000000, 0.013426, 0.000000, 0.012771,
#                                0.00000, 0.012177, 0.000000, 0.011636, 0.000000,
#                                0.00000, 0.000000, 0.010686, 0.000000, 0.010267]
        self.start_phi_layer = [0.00000, 0.015867, 0.000000, 0.000000, 0.000000,
                                0.00000, 0.014960, 0.014960, 0.000000, 0.000000,
                                0.00000, 0.000000, 0.000000, 0.000000, 0.000000,
                                0.00000, 0.000000, 0.000000, 0.000000, 0.000000]
        self.d_phi_layer = self._calculated_d_phi()
        self.lookup_table = self._prepare_wires_lookup()
        self.radii_table = self._prepare_wire_radius_lookup()
        self.angles_table = self._prepare_wire_angles_lookup()
        self.neighbours_table = self.get_neighbours()

    @property
    def n_events(self):
        return len(self.hits_data)

    def _get_first_wire(self):
        first_wire = np.zeros(20, dtype = int)
        for i in range(len(self.n_wires_in_layers)):
            first_wire[i] = sum(self.n_wires_in_layers[:i])
        return first_wire

    def _prepare_wires_lookup(self):
        """
        Prepares lookup table to map from [layer_id, cell_id] -> wire_id
        First 198 and last 306 wires not used currently
        :return:
        """
        lookup = np.zeros([len(self.n_wires_in_layers), max(self.n_wires_in_layers)], dtype='int')
        lookup[:, :] = - 1
        wire_id = 0
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            for cell_id in range(layer_size):
                lookup[layer_id, cell_id] = wire_id
                wire_id += 1
        assert wire_id == sum(self.n_wires_in_layers)
        return lookup

    def _prepare_wire_radius_lookup(self):
        """
        Prepares lookup table to map from wire id to the radial position
        :return: numpy.array of shape [total_wires]
        """
        first_wire = 0
        radii = np.zeros(self.total_wires, dtype=float)
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            radii[first_wire:first_wire + layer_size] = self.r_layers[layer_id]
            first_wire += layer_size
        return radii

    def _prepare_wire_angles_lookup(self):
        """
        Prepares lookup table to map from wire id to the angular position
        :return: numpy.array of shape [total_wires]
        """
        angles = np.zeros(self.total_wires, dtype=float)
        first_wire = 0
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            for wire_index in range(layer_size):
                angles[first_wire + wire_index] = (self.start_phi_layer[layer_id]
                    + self.d_phi_layer[layer_id] * wire_index)
            first_wire += layer_size
        angles %= 2*math.pi
        return angles

    def _get_wire_ids(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event
        """
        event = self.hits_data[event_id]
        cell_ids = event["O_cellID"]
        # + 1 since first is insensetive, i.e. layer 0 in root file is layer 1
        # in this structure
        layer_ids = event["O_layerID"] + 1
        wire_ids = self.lookup_table[layer_ids, cell_ids]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0], cell_ids[wire_ids < 0])
        return wire_ids

    def _calculated_d_phi(self):
        """
        Returns the phi separation of the wires as defined by the number of
        wires in the layer
        """
        return 2*math.pi/np.asarray(self.n_wires_in_layers)

    def get_measurement(self, event_id, name):
        """
        Returns requested measurement in all wires in requested event
        :return: numpy.array of shape [total_wires]
        """
        event = self.hits_data[event_id]
        wire_ids = self._get_wire_ids(event_id)
        measurement = event[name]
        result = np.zeros(self.total_wires, dtype=float)
        result[wire_ids] += measurement
        return result

    def get_energy_deposits(self, event_id):
        """
        Returns energy deposit in all wires
        :return: numpy.array of shape [total_wires]
        """
        energy_deposit = self.get_measurement(event_id, "O_edep")
        return energy_deposit

    def get_hit_types(self, event_id):
        """
        Returns hit type in all wires, where signal is 1, background is 2,
        nothing is 0
        :return: numpy.array of shape [total_wires]
        """
        event = self.hits_data[event_id]
        wire_ids = self._get_wire_ids(event_id)
        measurement = event["type"]
        coding = [1, 2, 2, 2]
        # Maps signal to 1, background to 2, and nothing to 0
        measurement = np.take(coding, measurement)
        result = np.zeros(self.total_wires, dtype=int)
        result[wire_ids] += measurement
        return result.astype(int)

    def get_neighbours(self):
        """
        Returns a sparse array of neighbour relations
        :return: scipy.sparse of shape [total_wires,total_wires]
        """
        neighbours = lil_matrix((self.total_wires, self.total_wires))
        first_wire = 0
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            for wire_index in range(layer_size):
                this_wire = wire_index + first_wire
                next_wire = first_wire + (wire_index + 1)%layer_size
                neighbours[next_wire, this_wire] = 1  # Clockwise
                neighbours[this_wire, next_wire] = 1  # Anti-Clockwise
                if (layer_id != len(self.n_wires_in_layers) - 1) :
                    wire_a = self.angles_table[this_wire]
                    angle_win = self.d_phi_layer[layer_id + 1] * 1.5
                    above = np.where(
                         (self.radii_table == self.r_layers[layer_id + 1]) &
                         ((abs(wire_a - self.angles_table) < angle_win) |
                         (2*math.pi - abs(wire_a - self.angles_table) < angle_win)
                         ))[0]
                    neighbours[this_wire, above[:]] = 1 # Above
                if (layer_id != 0):
                    wire_a = self.angles_table[this_wire]
                    angle_win = self.d_phi_layer[layer_id - 1] * 1.5
                    below = np.where(
                         (self.radii_table == self.r_layers[layer_id - 1]) &
                         ((abs(wire_a - self.angles_table) < angle_win) |
                         (2*math.pi - abs(wire_a - self.angles_table) < angle_win)
                         ))[0]
                    neighbours[this_wire, below[:]] = 1 # Below
            first_wire += layer_size
        return neighbours

    def get_wires_rhos_and_phis(self):
        """ Returns the positions of each wire in radial system

        :return: pair of numpy.arrays of shape [n_wires],
         - first one contains rho`s (radii)
         - second one contains phi's (angles)
        """
        return self.radii_table, self.angles_table

    def get_wires_xs_and_ys(self):
        """ Returns the positions of each wire in cartesian system

        :return: pair of numpy.arrays of shape [n_wires],
         - first one contains x`s
         - second one contains y's
        """
        x = self.radii_table * np.cos(self.angles_table)
        y = self.radii_table * np.sin(self.angles_table)
        return x, y
