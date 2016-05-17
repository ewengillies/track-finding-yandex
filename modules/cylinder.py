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

class CylindricalArray(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)
    def __ne__(self, other):
        return not self.__eq__(other)
    def __init__(self, n_by_layer, r_by_layer, phi0_by_layer):
        """
        This defines a cylindrical array of points from a layers.  It returns a
        flat enumerator of the points in the array, as well as pairwise
        distances between all points, and the neighbours of each point.  It also
        stores the position in both cartesian and polar coordinates of each
        point.

        :param n_by_layer: list of number of points by layer, sorted by radii of
                           corresponding layer
        :param r_by_layer: list of radii of each layer, sorted by radii
        :param phi0_by_layer: angular displacement of the first point of each
                              layer

        """
        self.n_by_layer = np.array(n_by_layer)
        self.r_by_layer = r_by_layer
        self.phi0_by_layer = phi0_by_layer
        self.n_points = sum(self.n_by_layer)

        self.first_point = self._get_first_point()
        self.dphi_by_layer = self._prepare_dphi_by_layer()
        self.point_lookup = self._prepare_points_lookup()
        self.point_rhos = self._prepare_point_rho()
        self.point_layers = np.repeat(np.arange(self.n_by_layer.size),
                                          self.n_by_layer) 
        self.point_indexes = np.arange(self.n_points) -
                             self.first_point[self.point_layers]
        self.point_phis = self._prepare_point_phi()
        self.point_x, self.point_y = self._prepare_point_cartesian()
        self.point_pol = self._prepare_polarity()
        self.point_dists = self._prepare_point_distances()
        self.point_neighbours, self.lr_neighbours = \
            self._prepare_point_neighbours()

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
                                              + self.dphi_by_layer[lay]*point)
        angles %= 2 * math.pi
        return angles

    def _prepare_point_cartesian(self):
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
        # All neighbours
        neigh = lil_matrix((self.n_points, self.n_points))
        # Only left and right neighbours
        lr_neigh = lil_matrix((self.n_points, self.n_points))
        # Loop over all layers
        for lay, n_points in enumerate(self.n_by_layer):
            # Define adjacent layers, noting outer most layers only have one
            # adjacent layer
            if lay == 0:
                adjacent_layers = [lay + 1]
            elif lay == len(self.n_by_layer) - 1:
                adjacent_layers = [lay - 1]
            else:
                adjacent_layers = [lay - 1, lay + 1]
            # Loop over points in current layer
            for point_index in range(n_points):
                # Define current wire, and wire counter-clockwise of this wire
                point = point_index + self.first_point[lay]
                nxt_point = (point_index + 1) % n_points + self.first_point[lay]
                # Define reciprocal neighbour relations on current layer
                neigh[nxt_point, point] = 1  # Clockwise
                lr_neigh[nxt_point, point] = 1
                neigh[point, nxt_point] = 1  # Anti-Clockwise
                lr_neigh[point, nxt_point] = 1
                # Define neighbour relations for adjacent layers
                # Start by finding position of point on layer (circle) as a
                # fraction
                rel_pos = self.point_phis[point] / (2 * math.pi)
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
                    nxt_a_point = (a_point + 1) % a_n_points
                    prv_a_point = (a_point - 1) % a_n_points
                    # Find the point_id for these three points
                    a_point += a_first
                    nxt_a_point += a_first
                    prv_a_point += a_first
                    # Define neighbour relations for points in adjacent layers
                    neigh[point, a_point] = 1  # Above/Below
                    neigh[point, nxt_a_point] = 1  # Above/Below Clockwise
                    neigh[point, prv_a_point] = 1  # Above/Below Anti-Clockwise
        return neigh.tocsr(), lr_neigh.tocsr()

    def _prepare_dphi_by_layer(self):
        """
        Returns the phi separation of the points as defined by the number of
        points in the layer
        """
        return 2 * math.pi / np.asarray(self.n_by_layer)

    def _prepare_polarity(self):
        """
        Prepare a table to record if the point is on an even or an odd layer,
        where the inner most layer is the zeroth layer.  0 value denotes even
        layer, 1 value denotes odd layer

        :return: numpy.array of shape [n_points] whose value is 1 for an odd
                 layer, 0 for an even layer
        """
        point_0 = self.first_point
        polarity = np.zeros(self.n_points, dtype=float)
        for lay, size in enumerate(self.n_by_layer):
            polarity[point_0[lay]:point_0[lay] + size] = lay % 2
        return polarity

    def get_neighbours(self, point_id):
        """
        Returns the neighbours of point_id as a list

        :return: list of neighbours of point_id
        """
        neighs = find(self.point_neighbours[point_id, :])[1]
        return neighs

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

    def get_layers(self, point_id):
        """
        Returns the layer index of a given point_id

        :return: Index of layer where point_id is
        """
        return self.point_layers[point_id]

    def get_indexes(self, point_id):
        """
        Returns the point index of a given point_id

        :return: Index of layer where point_id is
        """
        return self.point_indexes[point_id]

    def shift_wires(self, shift_size, point_id=np.arange(self.n_points)):
        """
        Get the index of the wire that is displaced from point_id by
        shift_size points counter clockwise in the same layer,
        respecting periodicity.

        :return: index of point shift_size  counter clockwise of point_id
        """
        layer = self.get_layers(point_id)
        index = self.get_indexes(point_id)
        index += shift_size
        index %= self.n_by_layer[layer]
        new_point = self.point_lookup[layer, index]
        return new_point

    def rotate_wire(self, point_id, shift_frac):
        """
        Get the index of the wire that is displaced from point_id by
        shift_frac of a revolution counter clockwise in the same layer,
        respecting periodicity.
        :param shift_frac: from [0, 1], 1 is complete rotation

        :return: index of point n_points_in_layer*shft_frac points
                 counter clockwise of point_id
        """
        layer = self.get_layers(point_id)
        index = self.get_indexes(point_id)
        n_points_in_layer = self.n_by_layer[layer]
        shift_size = int(round(shift_frac * (n_points_in_layer-1)))
        index += shift_size
        index %= n_points_in_layer
        new_point = self.point_lookup[layer, index]
        return new_point


class CyDet(CylindricalArray):
    def __init__(self, projection=0.5):
        """
        Defines the Cylindrical Detector Geometry
        """
        # Number of wires in each layer
        cydet_wires = [198, 204, 210, 216, 222, 228, 234, 240, 246,
                       252, 258, 264, 270, 276, 282, 288, 294, 300] 
        # Radius at end plate
        cydet_radii = [53.0, 54.6, 56.2, 57.8, 59.4, 61.0, 62.6, 64.2, 65.8,
                       67.4, 69.0, 70.6, 72.2, 73.8, 75.4, 77.0, 78.6, 80.2]
        # Phi0 at end plate
        cydet_phi0 = [0.015867, 0.015400, 0.000000, 0.014544, 0.00000, 0.000000,
                      0.013426, 0.000000, 0.012771, 0.00000, 0.012177, 0.000000,
                      0.011636, 0.000000, 0.00000, 0.000000, 0.010686, 0.000000]
        # Define the maximum angular shift of the wires in each layer from end
        # plate to the next  
        self.phi_shft = np.array([-0.190400, 0.184800, -0.179520, 0.174533,
                                  -0.169816, 0.165347, -0.161107, 0.157080, 
                                  -0.153248, 0.149600, -0.146121, 0.142800, 
                                  -0.139626, 0.136591, -0.155966, 0.152716,
                                  -0.149600, 0.146608])

        dphi_from_phi0 = self.theta_at_rel_z(projection)
        new_radius = self.radius_at_theta(cydet_radii, dphi_from_phi0)
        new_dphi = dphi_from_phi0 + cydet_phi0

        # Build the cylindrical array
        CylindricalArray.__init__(self, cydet_wires, new_radius, new_dphi)

    def theta_at_rel_z(self, z_dist, total_z=1.0):
        """
        Get the angular displacement of the wires in each layer as a function of
        the relative z_distance traversed

        :param z_dist:   z distance down CyDet volume
        :param total_z:  total z distance of CyDet volume

        :return: numpy array of angular shifts, one per layer
        """
        mid_val = self.phi_shft/2.
        this_shft = np.arctan(np.tan(self.phi_shft/2.)*(1. - 2.*z_dist/total_z))
        return mid_val - this_shft

    def rel_z_at_theta(self, theta, layer=0):
        """
        Get the relative z displacement for a given angular disagreement in a
        given layer

        :param theta: the angular displacement
        :param layer: the layer of disagreement

        :return: the relative z displacement from this stereometry
        """
        this_phi_shft = self.phi_shft[layer]/2.
        tan_phi_shft = np.tan(this_phi_shft)
        diff_phi_shift = np.tan(this_phi_shft - theta)
        return (tan_phi_shft - diff_phi_shift)/(2. * tan_phi_shft)

    def radius_at_theta(self, radius, this_theta):
        """
        Get the new radial distance of a wire as a function of :
          * The total angular displacement of the wire-hole between end plates
          * The angle displacement subtended
        
        Note the angular displacment subtended must be less than the total
        angular displacement by definition.  
        """
        assert not np.any(np.abs(this_theta) > np.abs(self.phi_shft)),\
            "The input angle is larger than the absoulte angular difference\n"+\
            "Abs. Diff. {} \n".format(self.phi_shft)+\
            "Reqs. Ang. {} \n".format(this_theta)
        return abs(radius*np.cos(self.phi_shft/2.)/np.cos(self.phi_shft/2. - this_theta))
        

class CTH(CylindricalArray):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, left_handed=False):
        """
        Defines the CTH geometry.  Note Cherenkov counter and light guide read
        out to the same volume here
        """
        self.left_handed = left_handed
        self.n_crystals = 64
        # The first two rows are for the active volumes, third is for excluded
        # volumes
        cth_n_vols = [self.n_crystals, self.n_crystals, self.n_crystals]
        cth_radii = [44.8, 48.5, 0.0]
        cth_phi0 = [-90., -90., -90.]
        CylindricalArray.__init__(self, cth_n_vols, cth_radii, cth_phi0)

        # Store the active volume names
        self.chrn_name = "TriChe"
        self.chlg_name = "TriCheL"
        self.scnt_name = "TriSci"
        self.active_names = [self.chrn_name, self.chlg_name, self.scnt_name]

        # Store the passive volume names
        self.chpm_name = "TriCheP"
        self.scpm_name = "TriSciP"
        self.sclg_name = "TriSciL"
        self.passive_names = [self.chpm_name, self.scpm_name, self.sclg_name]

        # Map volume names to row indexes
        self.name_to_row = dict()
        # Cherenkov volumes to inner ring
        self.name_to_row[self.chrn_name] = 0
        # Map the cherenkov light guide volumes to the cherenkov volumes
        self.name_to_row[self.chlg_name] =\
                self.name_to_row[self.chrn_name]
        # Map the scintillator to the outer ring
        self.name_to_row[self.scnt_name] = 1
        # Map the passive volumes to the passive row index
        for vol in self.passive_names:
            self.name_to_row[vol] = 2
        # Position to column
        self.pos_to_col = dict()
        self.pos_to_col['U'] = 1
        self.pos_to_col['D'] = 0

    def _prepare_dphi_by_layer(self):
        """
        Returns the phi separation of the points as defined by the number of
        points in the layer
        """
        if self.left_handed:
            return -2 * math.pi / np.asarray(self.n_by_layer)
        else:
            return 2 * math.pi / np.asarray(self.n_by_layer)

class TrackCenters(CylindricalArray):
    def __init__(self, r_min=10., r_max=50., rho_bins=10, arc_bins=0):
        """
        Defines the geometry of the centers of the potential tracks used in the
        Hough transform.  It is constructed from a minimum radius, maximum
        radius, number of radial layers, and spacial resolution in phi for all
        layers. The outer most layer will be at the maximal radius, while the
        inner most will be at the inner most radius.

        :param r_min:     Radius of inner most layer
        :param r_max:     Radius of outer most layer
        :param rho_bins:  Number of radial layers
        :param arc_bins:  Arc length between points along the layers. Default
                          value set this to be the same as the distance between
                          layers
        """
        # Define distance between layers to that the radii fall in [r_min,
        # r_max] inclusive
        drho = (r_max - r_min) / (rho_bins - 1)
        r_track_cent = [r_min + drho * n for n in range(rho_bins)]
        # Set default spacial resolution along layers to be the same as the
        # resolution between layers
        if arc_bins == 0:
            arc_res = drho
        else:
            arc_res = 2 * math.pi * r_min / arc_bins
        n_track_cent = [int(round(2 * math.pi * r_track_cent[n] / arc_res))
                        for n in range(rho_bins)]
        phi0_track_cent = [0] * rho_bins
        CylindricalArray.__init__(self, n_track_cent, r_track_cent, \
                phi0_track_cent)
