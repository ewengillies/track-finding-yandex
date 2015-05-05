import numpy as np
from scipy.stats import norm
from root_numpy import root2array
import math
from scipy.sparse import lil_matrix

"""
Notation used below:
 - wire_id is flat enumerator of all wires (from 0 to 4985)
 - layer_id is the index of layer (from 0 to 19)
 - cell_id is the index of wire in the layer (from 0 to layer_size -1)
"""


class Dataset:
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=old-style-class
    # pylint: disable=bad-continuation
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
        self.wires_by_layer = [198, 198, 204, 210, 216, 222, 228, 234, 240, 246,
                               252, 258, 264, 270, 276, 282, 288, 294, 300, 306]
        self.r_layers = [51.4, 53, 54.6, 56.2, 57.8, 59.4, 61, 62.6, 64.2, 65.8,
                         67.4, 69, 70.6, 72.2, 73.8, 75.4, 77, 78.6, 80.2, 81.8]
#        self.phi0_by_layer = [0.00000, 0.015867, 0.015400, 0.000000, 0.014544,
#                              0.00000, 0.000000, 0.013426, 0.000000, 0.012771,
#                              0.00000, 0.012177, 0.000000, 0.011636, 0.000000,
#                              0.00000, 0.000000, 0.010686, 0.000000, 0.010267]
        self.phi0_by_layer = [0.00000, 0.015867, 0.000000, 0.000000, 0.000000,
                              0.00000, 0.014960, 0.014960, 0.000000, 0.000000,
                              0.00000, 0.000000, 0.000000, 0.000000, 0.000000,
                              0.00000, 0.000000, 0.000000, 0.000000, 0.000000]
        self.first_wire = self._get_first_wire()
        self.total_wires = 4986
        assert sum(self.wires_by_layer) == self.total_wires

        self.sig_rho = 30. # determined from truth distribution of radial
                            # coordinates of hits
        self.sig_rho_sigma = 2 # defines the spread of the smearing of the
                                # signal track from the constant value
        self.trgt_rho = 20. # defined to cover the entire sense volume
        self.sig_trk_smear = 5   # defines the number of cells the track
                               # correspondence function will use when
                               # calculating probabilites
        self.trk_phi_bins = 20
        self.trk_rho_bins = 5
        self.trk_bins = self.trk_phi_bins*self.trk_rho_bins

        self.dphi_by_layer = self._calculated_d_phi()
        self.wire_lookup = self._prepare_wires_lookup()
        self.wire_rhos = self._prepare_wire_rho()
        self.wire_phis = self._prepare_wire_phi()
        self.wire_neighbours = self._prepare_wire_neighbours()

        self.track_lookup = self._prepare_track_lookup()
        self.track_phis = self._prepare_track_phis()
        self.track_rhos = self._prepare_track_rhos()
        self.correspondence = self._prepare_wire_track_corresp()

    @property
    def n_events(self):
        return len(self.hits_data)

    def _get_first_wire(self):
        """
        Returns a list of the indices of the first wire in each layer
        """
        first_wire = np.zeros(len(self.wires_by_layer), dtype=int)
        for i in range(len(self.wires_by_layer)):
            first_wire[i] = sum(self.wires_by_layer[:i])
        return first_wire

    def _prepare_wires_lookup(self):
        """
        Prepares lookup table to map from [layer_id, cell_id] -> wire_id
        First 198 and last 306 wires not used currently
        :return:
        """
        lookup = np.zeros([len(self.wires_by_layer),
                           max(self.wires_by_layer)], dtype='int')
        lookup[:, :] = - 1
        wire_id = 0
        for layer_id, layer_size in enumerate(self.wires_by_layer):
            for cell_id in range(layer_size):
                lookup[layer_id, cell_id] = wire_id
                wire_id += 1
        assert wire_id == sum(self.wires_by_layer)
        return lookup

    def _prepare_wire_rho(self):
        """
        Prepares lookup table to map from wire id to the radial position
        :return: numpy.array of shape [total_wires]
        """
        wire_0 = self.first_wire
        radii = np.zeros(self.total_wires, dtype=float)
        for lay, size in enumerate(self.wires_by_layer):
            radii[wire_0[lay]:wire_0[lay] + size] = self.r_layers[lay]
        return radii

    def _prepare_wire_phi(self):
        """
        Prepares lookup table to map from wire id to the angular position
        :return: numpy.array of shape [total_wires]
        """
        angles = np.zeros(self.total_wires, dtype=float)
        wire_0 = self.first_wire
        for lay, layer_size in enumerate(self.wires_by_layer):
            for wire in range(layer_size):
                angles[wire_0[lay] + wire] = (self.phi0_by_layer[lay]
                                              + self.dphi_by_layer[lay] * wire)
        angles %= 2*math.pi
        return angles

    def _prepare_wire_neighbours(self):
        """
        Returns a sparse array of neighbour relations
        :return: scipy.sparse of shape [total_wires,total_wires]
        """
        neighbours = lil_matrix((self.total_wires, self.total_wires))
        first_wire = 0
        for lay, layer_size in enumerate(self.wires_by_layer):
            for wire_index in range(layer_size):
                this_wire = wire_index + first_wire
                next_wire = first_wire + (wire_index + 1)%layer_size
                neighbours[next_wire, this_wire] = 1  # Clockwise
                neighbours[this_wire, next_wire] = 1  # Anti-Clockwise
                if lay != len(self.wires_by_layer) - 1:
                    wire_a = self.wire_phis[this_wire]
                    angle_win = self.dphi_by_layer[lay + 1] * 1.5
                    above = np.where(
                         (self.wire_rhos == self.r_layers[lay + 1]) &
                         ((abs(wire_a - self.wire_phis) < angle_win) |
                         (2*math.pi - abs(wire_a - self.wire_phis) < angle_win)
                         ))[0]
                    neighbours[this_wire, above[:]] = 1 # Above
                if lay != 0:
                    wire_a = self.wire_phis[this_wire]
                    angle_win = self.dphi_by_layer[lay - 1] * 1.5
                    below = np.where(
                         (self.wire_rhos == self.r_layers[lay - 1]) &
                         ((abs(wire_a - self.wire_phis) < angle_win) |
                         (2*math.pi - abs(wire_a - self.wire_phis) < angle_win)
                         ))[0]
                    neighbours[this_wire, below[:]] = 1 # Below
            first_wire += layer_size
        return neighbours

    def _get_wire_ids(self, event_id):
        """
        Returns the sequence of wire_ids that register hits in given event
        """
        event = self.hits_data[event_id]
        cell_ids = event["CdcCell_cellID"]
        # + 1 since first is insensetive, i.e. layer 0 in root file is layer 1
        # in this structure
        layer_ids = event["CdcCell_layerID"] + 1
        wire_ids = self.wire_lookup[layer_ids, cell_ids]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0],
                                                 cell_ids[wire_ids < 0])
        return wire_ids

    def _calculated_d_phi(self):
        """
        Returns the phi separation of the wires as defined by the number of
        wires in the layer
        """
        return 2*math.pi/np.asarray(self.wires_by_layer)

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

    def get_wires_rhos_and_phis(self):
        """
        Returns the positions of each wire in radial system

        :return: pair of numpy.arrays of shape [n_wires],
         - first one contains rho`s (radii)
         - second one contains phi's (angles)
        """
        return self.wire_rhos, self.wire_phis

    def get_wires_xs_and_ys(self):
        """
        Returns the positions of each wire in cartesian system

        :return: pair of numpy.arrays of shape [n_wires],
         - first one contains x`s
         - second one contains y's
        """
        x_coor = self.wire_rhos * np.cos(self.wire_phis)
        y_coor = self.wire_rhos * np.sin(self.wire_phis)
        return x_coor, y_coor

    def _prepare_track_lookup(self):
        """
        Prepares lookup table to map from [rho_bin, phi_bin] -> bin_id
        :return: numpy.array of shape [trk_bins]
        """
        track_lookup = np.zeros([self.trk_rho_bins,
                                 self.trk_phi_bins], dtype='int')
        track_lookup[:, :] = - 1
        track_bin = 0
        for rho_bin in range(self.trk_rho_bins):
            for phi_bin in range(self.trk_phi_bins):
                track_lookup[rho_bin, phi_bin] = track_bin
                track_bin += 1
        assert track_bin == self.trk_rho_bins*self.trk_phi_bins
        return track_lookup

    def _prepare_track_rhos(self):
        """
        Returns the physical locations of each track_bin in rho.

        Maximal distance is defined as the location where the signal track will
        enter the last layer.

        Minimal distance defined as the distance where the track+sig_trk_smear
        will enter the first layer, provided this distance is not less than the
        physics allows (i.e. track must pass through both target and detector
        region)

        :return: numpy.array of shape [trk_bins]
        """
        t_0 = 0
        track_rhos = np.zeros(self.trk_bins)
        r_max = self.r_layers[-2] - self.sig_rho
        r_min = max(self.sig_rho - self.trgt_rho,
                    self.r_layers[1] - self.sig_rho - self.sig_trk_smear)
        drho = (r_max - r_min)/(self.trk_rho_bins-1)
        for n_bin in range(self.trk_rho_bins):
            track_rhos[t_0:t_0 + self.trk_phi_bins] = r_min + drho*n_bin
            t_0 += self.trk_phi_bins
        return track_rhos

    def _prepare_track_phis(self):
        """
        Discretizes the possible locations of the center of a track in phi

        :return: numpy.array of shape [trk_bins], contains possible
        centers of the tracks in phi
        """
        dphi = (2*math.pi)/self.trk_phi_bins
        return np.fromfunction(lambda x: (x%self.trk_phi_bins)*dphi,
                               (self.trk_bins,))

    def get_tracks_rhos_and_phis(self):
        """
        Returns the positions of each track center

        :return: pair of numpy.arrays of shape [trk_bins],
         - first one contains rho`s (radii)
         - second one contains phi's (angles)
        """
        return self.track_rhos, self.track_phis

    def polar_dist(self, rho_1, phi_1, rho_2, phi_2):
        """
        Returns Euclidian distance between to points in polar coordinates
        """
        return np.sqrt(rho_1**2 + rho_2**2 - 2*rho_1*rho_2*np.cos(phi_1-phi_2))

    def dist_prob(self, distance):
        """
        Defines the probability distribution used for correspondence matrix
        """
        return norm.pdf(distance, scale=self.sig_rho_sigma)

    def trk_wire_dist(self, wr_id, t_bin):
        """
        Returns distance between a wire given by wr_id and the center of a
        potential track given by t_bin
        """
        return self.polar_dist(self.wire_rhos[wr_id], self.wire_phis[wr_id],
                               self.track_rhos[t_bin], self.track_phis[t_bin])

    def trk_wire_prob(self, wr_id, t_bin):
        """
        Returns probability that hit on wire belongs to given track bin
        """
        return norm.pdf(self.trk_wire_dist(wr_id, t_bin) - self.sig_rho,
                        scale=self.sig_rho_sigma)

    def _prepare_wire_track_corresp(self):
        """
        Defines the probability that a given wire belongs to a track centered at
        a given track center bin
        :returns: scipy.sparse matrix of shape [n_wires, n_track_bin]
        """
        corresp = lil_matrix((self.total_wires, self.trk_bins))
        for trk_bin in range(self.trk_bins):
            for wire_id in range(self.total_wires):
                this_dist = self.trk_wire_dist(wire_id, trk_bin) - self.sig_rho
                if abs(this_dist) < self.sig_trk_smear:
                    corresp[wire_id, trk_bin] = self.dist_prob(this_dist)
        return corresp
