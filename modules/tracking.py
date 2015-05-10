import numpy as np
from scipy.stats import norm
#import math
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
from cylinder import TrackCenters

"""
Notation used below:
 - wire_id is flat enumerator of all wires (from 0 to 4985)
 - layer_id is the index of layer (from 0 to 19)
 - cell_id is the index of wire in the layer (from 0 to layer_size -1)
"""


class Hough(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self, hit_data, sig_rho=30., sig_rho_sgma=2., sig_rho_smear=5.,
                 trgt_rho=20.):
        """
        This class resprsents a hough transform method. It initiates from a data
        file, and over lays a track center geometry on this.  It also defines a
        signal track radius. The track center geometry is defined so that only
        track centers that yield signal tracks that pass through the CyDet and
        the target are considered. The signal radius and target radius are
        defined by default to produce tracks whose maximal rho value can lie
        anywhere between the first and the last layer.

        This class calculates the probability that a given hit point in the
        CyDet belongs to a track centered at a given point in the TrackCenters
        geometry.  This probability peaks at the distance sig_rho, and decays
        with decay width sig_rho_sgma.  The probability is only defined for
        CyDet wires that are within sig_rho_smear of the signal track distance
        from the given center.  This means for track center trck_id and CyDet
        wire_id, wires that satisfy

        distance(trck_id, wire_id) - sig_rho > sig_rho_smear

        are included in the probability calculations.

        :param sig_rho: ideal signal track radius
        :param sig_rho_sgma: decay length of the hit point to track probability
        :paran sig_rho_smear: distance from the signal radius for a given track
                              center that defines the cells of interest
        :param trgt_rho: radius of target.  Note: may be non-phyiscal, it
                         represents the constraint that the track started near
                         the origin.
        """

        self.hit_data = hit_data
        self.sig_rho = sig_rho
        self.sig_rho_sgma = sig_rho_sgma
        self.trgt_rho = trgt_rho
        self.sig_trk_smear = sig_rho_smear

        # Set the geometry of the TrackCenters to cover regions where the signal
        # track passes through the target and the CyDet volume.  Specifically,
        # enforce that the track's outer most hits may lie in the first or last
        # layer.
        r_max = self.hit_data.cydet.r_by_layer[-2] - self.sig_rho
        r_min = max(self.sig_rho - self.trgt_rho,
                    self.hit_data.cydet.r_by_layer[1]
                        - self.sig_rho - self.sig_trk_smear)
        self.track = TrackCenters(rho_bins=10, r_min=r_min, r_max=r_max)

        self.track_wire_dists = self._prepare_track_distances()
        self.correspondence = self._prepare_wire_track_corresp()

    def _prepare_track_distances(self):
        """
        Returns a numpy array of distances between tracks and wires

        :return: numpy array of shape [n_wires,n_tracks]
        """
        wire_xy = np.column_stack((self.hit_data.cydet.point_x,
                                   self.hit_data.cydet.point_y))
        trck_xy = np.column_stack((self.track.point_x, self.track.point_y))
        distances = cdist(wire_xy, trck_xy)
        return distances

    def dist_prob(self, distance):
        """
        Defines the probability distribution used for correspondence matrix

        :return: Gaussian of distance
        """
        return norm.pdf(distance, scale=self.sig_rho_sgma)

    def _prepare_wire_track_corresp(self):
        """
        Defines the probability that a given wire belongs to a track centered at
        a given track center bin

        :returns: scipy.sparse matrix of shape [n_wires, n_track_bin]
        """
        corsp = lil_matrix((self.hit_data.cydet.n_points, self.track.n_points))
        # Loop over all track centers
        for trck in range(self.track.n_points):
            # Loop over all wires in CyDet
            for wire in range(self.hit_data.cydet.n_points):
                # Calculate how far the wire is from the signal track centered
                # at the current track center
                this_dist = self.track_wire_dists[wire, trck] - self.sig_rho
                # If its within tolerance, return a probability
                if abs(this_dist) < self.sig_trk_smear:
                    corsp[wire, trck] = self.dist_prob(this_dist)
        return corsp

    def find_track(self):
        print "I found a track!"
