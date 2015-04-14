import numpy as np
from root_numpy import root2array

"""
Notation used below:
 - wire_id is flat enumerator of all wires (from 0 to 4985)
 - layer_id is the index of layer (from 0 to 19)
 - cell_id is the index of wire in the layer (from 0 to layer_size -1)


"""


class Dataset:
    def __init__(self, path="data/signal_TDR.root", treename='tree'):
        """Dataset provides an interface to work with MC stored in root format.
        Results of methods are either numpy.arrays or scipy.sparse objects.

        :param path: path to rootfile
        :param treename: name of the tree in root dataset
        """
        self.hits_data = root2array(path, treename=treename)
        # Hardcode information about wires in the CDC
        self.n_wires_in_layers = [198, 198, 204, 210, 216, 222, 228, 234, 240, 246,
                                  252, 258, 264, 270, 276, 282, 288, 294, 300, 306]
        self.total_wires = 4986
        assert sum(self.n_wires_in_layers) == self.total_wires

        self.lookup_table = self._prepare_wires_lookup()

    def _prepare_wires_lookup(self):
        """
        Preparing lookup table [layer_id, cell_id] -> wire_id
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

    def _get_wire_ids(self, event_id):
        """Returns the sequence of wire_ids activated in event passed to function"""
        event = self.hits_data[event_id]
        cell_ids = event["CdcCell_cellID"]
        # + 1 since these are only sensitive layers
        layer_ids = event["CdcCell_layerID"] + 1
        wire_ids = self.lookup_table[layer_ids, cell_ids]
        assert np.all(wire_ids >= 0), \
            'Wrong id of wire here {} {}'.format(layer_ids[wire_ids < 0], cell_ids[wire_ids < 0])
        return wire_ids

    def get_energy_deposits(self, event_id):
        """Returns energy deposit in all wires
        :return: numpy.array of shape [total_wires]
        """
        event = self.hits_data[event_id]
        wire_ids = self._get_wire_ids(event_id)
        energy_deposits = event["CdcCell_edep"]
        result = np.zeros(self.total_wires, dtype=float)
        result[wire_ids] += energy_deposits
        return result

