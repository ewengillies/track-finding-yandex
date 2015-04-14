import numpy as np
from root_numpy import root2array

## Import the root file as a numpy array
HIT_DATA = root2array("../../tracking_data/signal_TDR.root", 'tree')
## Hardcode information about wires in the CDC
## Total number of wires is 4986
NWIRES = [198, 198, 204, 210, 216, 222, 228, 234, 240, 246,
          252, 258, 264, 270, 276, 282, 288, 294, 300, 306]

## Access functions
def get_wire_index(layer, wire):
    """
    Returns global index of wire given layer and position in layer
    """
    return sum(NWIRES[:layer]) + wire

def get_wire_layer(index):
    """
    Returns the layer and position in layer given global index
    """
    for i in range(0, 17):
        if index > NWIRES[i]:
            index -= NWIRES[i]
        else:
            return {'layer':i, 'index':index}

def get_wire_data(event, layer, wire):
    """
    Returns index of wire in the given array
    """
    both_filter = np.where((HIT_DATA[event]["CdcCell_cellID"] == wire) &
                           (HIT_DATA[event]["CdcCell_layerID"] == layer))
    return both_filter[0][0]

def get_hit_wire(event, index):
    """
    Returns the position of the  nth hit wire in the requested event
    """
    return {'layer':HIT_DATA[event]["CdcCell_layerID"][index],
            'wire' :HIT_DATA[event]["CdcCell_cellID"][index]}

def get_edeps(event):
    energy_deps = np.zeros(shape=sum(NWIRES))
    print energy_deps

## Test area
print get_hit_wire(1, 4)
WIRE_POS = get_hit_wire(1, 4)
print get_wire_index(WIRE_POS["layer"], WIRE_POS["wire"])

## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
    REP = ''
    while not REP in ['q', 'Q']:
        REP = raw_input('enter "q" to quit: ')
        if 1 < len(REP):
            REP = REP[0]
