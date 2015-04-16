import numpy as np
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
        self.n_wires_in_layers = [198,198,204,210,216,222,228,234,240,246,
                                  252,258,264,270,276,282,288,294,300,306]
        self.r_layers = [51.4, 53, 54.6, 56.2, 57.8, 59.4, 61, 62.6, 64.2, 65.8,
                         67.4, 69, 70.6, 72.2, 73.8, 75.4, 77, 78.6, 80.2, 81.8]
        self.start_phi_layer = [0.00000, 0.015867, 0.015400, 0.000000, 0.014544,
                                0.00000, 0.000000, 0.013426, 0.000000, 0.012771,
                                0.00000, 0.012177, 0.000000, 0.011636, 0.000000,
                                0.00000, 0.000000, 0.010686, 0.000000, 0.010267]

        self.total_wires = 4986
        assert sum(self.n_wires_in_layers) == self.total_wires
        self.lookup_table = self._prepare_wires_lookup()
        self.radii_table = self._prepare_wire_radius_lookup()
        self.angles_table = self._prepare_wire_angles_lookup()
        self.neighbours_table = self.get_neighbours()

    @property
    def n_events(self):
        return len(self.hits_data)

    def _prepare_wires_lookup(self):
        """
        Prepares lookup table to map from [layer_id, cell_id] -> wire_id
        First 198 and last 306 wires not used currently
        :return:
        """
        lookup = np.zeros([len(self.n_wires_in_layers),max(self.n_wires_in_layers)], dtype='int')
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
        Prepares lookup table to map from wire id to  the radial position
        :return: numpy.array of shape [total_wires]
        """
        first_wire = 0
        radii = np.zeros(self.total_wires, dtype=float)
        for layer_id, layer_size in enumerate(self.n_wires_in_layers):
            radii[first_wire:first_wire+layer_size] = self.r_layers[layer_id]
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
                angles[first_wire+wire_index] = self.start_phi_layer[layer_id] + 2*math.pi/layer_size*wire_index
            first_wire += layer_size
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
        Returns hit type in all wires
        :return: numpy.array of shape [total_wires]
        """
        event = self.hits_data[event_id]
        wire_ids = self._get_wire_ids(event_id)
        measurement = event["CdcCell_hittype"]
        coding = [1, 2, 2, 2]
        # Maps signal to 1, background to 2, and nothing to 0
        measurement = np.take(coding, measurement)
        result = np.zeros(self.total_wires, dtype=float)
        result[wire_ids] += measurement
        return result

    def get_neighbours(self):
        """
        Returns a sparse array of neighbour relations
        :return: scipy.sparse of shape [total_wires,total_wires]
        """
        neighbours = lil_matrix((self.total_wires, self.total_wires))
        first_wire = 0
        for layer_id, layer_size in enumerate(self.n_wires_in_layers[:18]):
            for wire_index in range(layer_size):
                this_wire = wire_index + first_wire
                neighbours[this_wire,this_wire+(wire_index+1)%layer_size] = 1 # Clockwise
                neighbours[this_wire,this_wire+(wire_index-1)%layer_size] = 1 # Anti-Clockwise
                above = np.where((abs(self.angles_table-self.angles_table[this_wire]) < 0.005) & (self.radii_table == self.r_layers[layer_id+1]))
                neighbours[this_wire,above[:]] = 1
            first_wire += layer_size
        return neighbours
