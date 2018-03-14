

from data import Dataset
import numpy as np
from scipy.sparse import find

__author__ = 'Alex Rogozhnikov'

signal = Dataset('data/signal_TDR.root', trk_phi_bins=10, trk_rho_bins=8)


def test_neighbors_counts():
    """
    Test the amount of neighbors
    """
    bad_neighs = np.zeros(signal.total_wires)
    for wire in range(signal.total_wires):
        # Count the neighbours
        n_neigh = len(find(signal.wire_neighbours[wire, :])[1])
        if (wire < signal.first_wire[1]) or (wire >= signal.first_wire[-1]):
            if n_neigh != 5:
                bad_neighs[wire] = n_neigh
        else:
            if n_neigh != 8:
                bad_neighs[wire] = n_neigh

    assert np.sum(bad_neighs) == 0, 'some bad wire exists!'


def test_neighbours_count():
    signal_wires = signal.wire_neighbours.sum(axis=1)
    bad_neighs = np.zeros(signal.total_wires)
    for wire in range(signal.total_wires):
        # Count the neighbours
        n_neigh = len(find(signal.wire_neighbours[wire, :])[1])
        if (wire < signal.first_wire[1]) or (wire >= signal.first_wire[-1]):
            if n_neigh != 5:
                bad_neighs[wire] = n_neigh
        else:
            if n_neigh != 8:
                bad_neighs[wire] = n_neigh

    assert np.sum(bad_neighs) == 0, 'some bad wire exists!'


def test_no_closer_neighbors():
    # Check that there are no cells closer then the current neighbours, keeping layer spacing in mind
    far_neighs = np.zeros(signal.total_wires)
    furthest_n = np.zeros(signal.total_wires)
    closest_nn = np.zeros(signal.total_wires)

    for wire in range(signal.total_wires):
        # Get the neighbours
        neighs = find(signal.wire_neighbours[wire, :])[1]
        # Find the maximal distance
        max_n_dist = max(signal.wire_dists[wire, neighs])
        # Invert to find wire responsible
        far_n = np.where(signal.wire_dists[wire, neighs] == max_n_dist)
        far_n = neighs[far_n[0][0]]
        # Find the next to nearest neighbours
        n_neighs = find(signal.wire_neighbours[neighs, :])[1]
        n_neighs = set(n_neighs) - set(neighs) - set([wire])
        n_neighs = list(n_neighs)
        # Find the closest one, record which wire this was
        min_nn_dist = min(signal.wire_dists[wire, n_neighs])
        close_nn = np.where(signal.wire_dists[wire, n_neighs] == min_nn_dist)
        close_nn = n_neighs[close_nn[0][0]]
        # Check the distance
        dist = max_n_dist - min_nn_dist
        # If this distance is less than zero and these wires are on the same layer, we're in trouble
        if (max_n_dist - min_nn_dist > 1e-9) and (signal.wire_rhos[far_n] == signal.wire_rhos[close_nn]):
            far_neighs[wire] = dist
            furthest_n[wire] = far_n
            closest_nn[wire] = close_nn

    assert np.sum(far_neighs != 0) == 0, 'There exist too far neighbors'
