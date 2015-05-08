import numpy as np
import math
from scipy.sparse import lil_matrix, find
from scipy.spatial.distance import pdist, squareform

"""
Notation used below:
 - point_id is flat enumerator of all points in the cylinder
 - layer_id is the index of layer
 - point_index is the index of point in the layer
"""

class Array(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __init__(self, n_by_layer, r_by_layer, phi0_by_layer):
        """
        This defines a cylindrical array of points from a layers.  It returns a
        flat enumerator of the points in the array, as well as pairwise
        distances between all points, and the neighbours of each point.  It also
        stores the position in both cartisian and polar coordinates of each
        point.

        :param n_by_layer: list of number of points by layer, sorted by radii of
                           corresponding layer
        :param r_by_layer: list of radii of each layer, sorted by radii
        :param phi0_by_layer: angular displacement of the first point of each
                              layer

        """
        self.n_by_layer = n_by_layer
        self.r_by_layer = r_by_layer
        self.phi0_by_layer = phi0_by_layer
        self.n_points = sum(self.n_by_layer)

        self.first_point = self._get_first_point()
        self.dphi_by_layer = self._prepare_dphi_by_layer()
        self.point_lookup = self._prepare_points_lookup()
        self.point_rhos = self._prepare_point_rho()
        self.point_phis = self._prepare_point_phi()
        self.point_x, self.point_y = self._prepare_point_cartisian()
        self.point_dists = self._prepare_point_distances()
        self.point_neighbours = self._prepare_point_neighbours()

    def _get_first_point(self):
        """
        Returns the point_id of the first point in each layer

        :return: numpy array of first point in each layer
        """
        first_point = np.zeros(len(self.n_by_layer), dtype=int)
        for i in range(len(self.n_by_layer)):
            first_point[i] = sum(self.n_by_layer[:i])
        return first_point

    def _prepare_points_lookup(self):
        """
        Prepares lookup table to map from [layer_id, point_index] -> point_id

        :return:
        """
        lookup = np.zeros([len(self.n_by_layer),
                           max(self.n_by_layer)], dtype='int')
        lookup[:, :] = - 1
        point_id = 0
        for layer_id, layer_size in enumerate(self.n_by_layer):
            for cell_id in range(layer_size):
                lookup[layer_id, cell_id] = point_id
                point_id += 1
        assert point_id == sum(self.n_by_layer)
        return lookup

    def _prepare_point_rho(self):
        """
        Prepares lookup table to map from point_id to the radial position

        :return: numpy.array of shape [n_points]
        """
        point_0 = self.first_point
        radii = np.zeros(self.n_points, dtype=float)
        for lay, size in enumerate(self.n_by_layer):
            radii[point_0[lay]:point_0[lay] + size] = self.r_by_layer[lay]
        return radii

    def _prepare_point_phi(self):
        """
        Prepares lookup table to map from point_id to the angular position

        :return: numpy.array of shape [n_points]
        """
        angles = np.zeros(self.n_points, dtype=float)
        point_0 = self.first_point
        for lay, layer_size in enumerate(self.n_by_layer):
            for point in range(layer_size):
                angles[point_0[lay] + point] = (self.phi0_by_layer[lay]
                                              + self.dphi_by_layer[lay] * point)
        angles %= 2*math.pi
        return angles

    def _prepare_point_cartisian(self):
        """
        Returns the positions of each point in cartesian system

        :return: pair of numpy.arrays of shape [n_points],
         - first one contains x coordinates
         - second one contains y coordinates
        """
        x_coor = self.point_rhos * np.cos(self.point_phis)
        y_coor = self.point_rhos * np.sin(self.point_phis)
        return x_coor, y_coor

    def _prepare_point_distances(self):
        """
        Returns a numpy array of distances between points

        :return: numpy array of shape [n_points,n_points]
        """
        point_xy = np.column_stack((self.point_x, self.point_y))
        distances = pdist(point_xy)
        return squareform(distances)

    def _prepare_point_neighbours(self):
        """
        Returns a sparse array of neighbour relations, where slicing should be
        done in the row index, i.e. find(neighbours[point_0,:]) will return the
        neighbours of point_0

        :return: scipy.sparse Compressed Sparse Row of shape
        [n_points,n_points]
        """
        neigh = lil_matrix((self.n_points, self.n_points))
        # Loop over all layers
        for lay, n_points in enumerate(self.n_by_layer):
            # Define adjacent layers, noting outer most layers only have one
            # adjacent layer
            if lay == 0:
                adjacent_layers = [lay+1]
            elif lay == len(self.n_by_layer) - 1:
                adjacent_layers = [lay-1]
            else:
                adjacent_layers = [lay-1, lay+1]
            # Loop over points in current layer
            for point_index in range(n_points):
                # Define current wire, and wire counter-clockwise of this wire
                point = point_index +  self.first_point[lay]
                nxt_point = (point_index + 1)%n_points + self.first_point[lay]
                # Define reciprocal neighbour relations on current layer
                neigh[nxt_point, point] = 1  # Clockwise
                neigh[point, nxt_point] = 1  # Anti-Clockwise
                # Define neighbour relations for adjacent layers
                # Start by finding position of point on layer (circle) as a
                # fraction
                rel_pos = self.point_phis[point]/(2*math.pi)
                # Loop over adjacent layers
                for a_lay in adjacent_layers:
                    # Set constants of adjacent layer
                    a_n_points = self.n_by_layer[a_lay]
                    a_first = self.first_point[a_lay]
                    # Find point in adjacent layer closest in phi to
                    # current point
                    # Account for phi0
                    a_point = rel_pos - (self.phi0_by_layer[a_lay]/(2*math.pi))
                    # Find index of adjacent layer point
                    a_point *= a_n_points
                    a_point = round(a_point)
                    # Enforce periodicity for boundary points
                    a_point %= a_n_points
                    # Find points clockwise and counter clockwise to the layer
                    # adjacent point
                    nxt_a_point = (a_point+1)%a_n_points
                    prv_a_point = (a_point-1)%a_n_points
                    # Find the point_id for these three points
                    a_point += a_first
                    nxt_a_point += a_first
                    prv_a_point += a_first
                    # Define neighbour relations for points in adjacent layers
                    neigh[point, a_point] = 1      # Above/Below
                    neigh[point, nxt_a_point] = 1  # Above/Below Clockwise
                    neigh[point, prv_a_point] = 1  # Above/Below Anti-Clockwise
        return neigh.tocsr()

    def get_neighbours(self, point_id):
        """
        Returns the neighbours of point_id as a list

        :return: list of neighbours of point_id
        """
        neighs = find(self.point_neighbours[point_id, :])[1]
        return neighs

    def _prepare_dphi_by_layer(self):
        """
        Returns the phi separation of the points as defined by the number of
        points in the layer
        """
        return 2*math.pi/np.asarray(self.n_by_layer)

    def get_points_rhos_and_phis(self):
        """
        Returns the positions of each point in radial system

        :return: pair of numpy.arrays of shape [n_points],
         - first one contains rho`s (radii)
         - second one contains phi's (angles)
        """
        return self.point_rhos, self.point_phis

    def get_points_xs_and_ys(self):
        """
        Returns the positions of each point in a cartisian system

        :return: pair of numpy.arrays of shape [n_points],
         - first one contains rho`s (radii)
         - second one contains phi's (angles)
        """
        return self.point_x, self.point_y

class CyDet(Array):
    def __init__(self):
        """
        Defines the Cylindrical Detector Geometry
        """
        cydet_wires = [198, 204, 210, 216, 222, 228, 234, 240, 246,
                       252, 258, 264, 270, 276, 282, 288, 294, 300]
        cydet_radii = [53, 54.6, 56.2, 57.8, 59.4, 61, 62.6, 64.2, 65.8,
                       67.4, 69, 70.6, 72.2, 73.8, 75.4, 77, 78.6, 80.2]
#        self.phi0_by_layer = [0.00000, 0.015867, 0.015400, 0.000000, 0.014544,
#                              0.00000, 0.000000, 0.013426, 0.000000, 0.012771,
#                              0.00000, 0.012177, 0.000000, 0.011636, 0.000000,
#                              0.00000, 0.000000, 0.010686, 0.000000, 0.010267]
        cydet_phi0 = [0.015867, 0.0, 0.0, 0.0, 0.0, 0.014960,
                      0.014960, 0.0, 0.0, 0.0, 0.0, 0.000000,
                      0.000000, 0.0, 0.0, 0.0, 0.0, 0.000000]
        Array.__init__(self, cydet_wires, cydet_radii, cydet_phi0)

class TrackCenters(Array):
    def __init__(self, r_min=10., r_max=50., rho_bins=10, arc_res=0):
        """
        Defines the geometry of the centers of the potential tracks used in the
        Hough transform.  It is constructed from a minimum radius, maximum
        radius, number of radial layers, and spacial resolution in phi for all
        layers.  The outer most layer will be at the maximal radius, while the
        inner most will be at the inner most radius.

        :param r_min: Radius of inner most layer
        :param r_max: Radius of outer most layer
        :param rho_bin: Number of radial layers
        :param arc_res: Arc length between points along the layers. Default
                        value set this to be the same as the distance between
                        layers
        """
        # Define distance between layers to that the radii fall in [r_min,
        # r_max] inclusive
        drho = (r_max - r_min)/(rho_bins-1)
        r_track_cent = [r_min + drho*n for n in range(rho_bins)]
        # Set default spacial resolution along layers to be the same as the
        # resolution between layers
        if arc_res == 0:
            arc_res = drho
        n_track_cent = [int(round(2*math.pi*r_track_cent[n]/arc_res))
                        for n in range(rho_bins)]
        phi0_track_cent = [0] * rho_bins
        Array.__init__(self, n_track_cent, r_track_cent, phi0_track_cent)
