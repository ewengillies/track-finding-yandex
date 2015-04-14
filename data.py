import numpy as np
from root_numpy import root2array

<<<<<<< HEAD
## Import the root file as a numpy array
HIT_DATA = root2array("../../tracking_data/signal_TDR.root", 'tree')
## Hardcode information about wires in the CDC
## Total number of wires is 4986
NWIRES = [198, 198, 204, 210, 216, 222, 228, 234, 240, 246,
          252, 258, 264, 270, 276, 282, 288, 294, 300, 306]
np.set_printoptions(threshold=np.nan)
## Objects
def empty_wire_array():
    return np.zeros(shape=sum(NWIRES))

## Access functions
## MAPPING FUNCTIONS
def get_wire_index(layer, wire):
    """
    Returns global index of wire given layer and position in layer
    """
    return sum(NWIRES[:layer]) + wire

def get_wire_layer(index):
    """
    Returns the layer and position in layer given global index
    """
    for layer in range(0, 19):
        if index > NWIRES[layer]:
            index -= NWIRES[layer]
        else:
            return {'layer':layer, 'wire':index}

def get_layers():
    """
    Returns a 1D wire array valued by layer
    """
    wire_by_layer = empty_wire_array()
    for i in range(0, 200):
        position = get_wire_layer(i)
        print position
        #wire_by_layer[i] = layer
    #print wire_by_layer

## Event wise accessors
def get_hit_wire_layer(event, index):
    """
    Returns the position of the  nth hit wire in the requested event
    """
    return {'layer':HIT_DATA[event]["CdcCell_layerID"][index],
            'wire' :HIT_DATA[event]["CdcCell_cellID"][index]}

def get_hit_wire_index(event, hit):
    """
    Returns the wire index of a given hit in the given event
    """
    hit_wire = get_hit_wire_layer(event, hit)
    return get_wire_index(hit_wire["layer"], hit_wire["wire"])

def get_wire_data(event, layer, wire):
    """
    Returns index of wire in the given array
    """
    both_filter = np.where((HIT_DATA[event]["CdcCell_cellID"] == wire) &
                           (HIT_DATA[event]["CdcCell_layerID"] == layer))
    return both_filter[0][0]

def get_deposited_energies(event):
    """
    Returns the 1D wire array by  deposited energies for the selected event
    """
    energy_deps = empty_wire_array()
    for hit in range(0, HIT_DATA[event]["CdcCell_nHits"]):
        wire_index = get_hit_wire_index(event, hit)
        energy_deps[wire_index] = HIT_DATA[event]["CdcCell_edep"][hit]
    return energy_deps

## Test area
print get_hit_wire_layer(1, 4)
WIRE_POS = get_hit_wire_layer(1, 4)
print get_wire_index(WIRE_POS["layer"], WIRE_POS["wire"])
get_layers()

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    REP = ''
    while not REP in ['q', 'Q']:
        REP = raw_input('enter "q" to quit: ')
        if 1 < len(REP):
            REP = REP[0]
=======
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

    @property
    def n_events(self):
        return len(self.hits_data)

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

>>>>>>> acf6648d68ec380dba93f064f04fe1e1cbc7ed7b
