import numpy as np
from root_numpy import root2array

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
