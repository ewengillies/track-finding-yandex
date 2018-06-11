import math
import numpy as np
from root_numpy import root2array
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


    def __init__(self, wire_x, wire_y, layerID):
        """
        This defines a cylindrical array of points read in from positional
        information, including IDs.

        It returns a flat enumerator of the points in the array, as well as
        pairwise distances between all points, and the neighbours of each point.
        It also stores the position in both cartesian and polar coordinates of
        each point.

        :param :

        """
        self.point_x = wire_x
        self.point_y = wire_y
        self.point_layers = layerID
        _, self.n_by_layer = np.unique(layerID, return_counts=True)
        self.first_point = self._get_first_point(self.n_by_layer)
        self.n_points = sum(self.n_by_layer)
        self.point_lookup = self._prepare_points_lookup(self.n_by_layer)
        self.point_layers = np.repeat(np.arange(self.n_by_layer.size),
                                          self.n_by_layer)
        self.point_indexes = np.arange(self.n_points) -\
                self.first_point[self.point_layers]
        self.dphi_by_layer = self._prepare_dphi_by_layer(self.n_by_layer)

        self.point_rhos = np.sqrt(np.square(wire_x)+np.square(wire_y))
        self.point_phis = np.arctan2(wire_y, wire_x)

        self.point_pol = self._prepare_polarity()
        self.point_dists = self._prepare_point_distances()
        self.point_neighbours, self.lr_neighbours = \
            self._prepare_point_neighbours(self.point_phis[self.first_point])

    def _old_constructor(self, in_n_by_layer, in_r_by_layer, in_phi0_by_layer):
        n_by_layer = np.array(in_n_by_layer)
        r_by_layer = np.array(in_r_by_layer)
        phi0_by_layer = np.array(in_phi0_by_layer)
        n_points = sum(n_by_layer)
        first_point = self._get_first_point(n_by_layer)
        dphi_by_layer = self._prepare_dphi_by_layer(n_by_layer)
        point_rhos = self._prepare_point_rho(first_point, n_points,
                                             n_by_layer, r_by_layer)
        point_layers = np.repeat(np.arange(n_by_layer.size), n_by_layer)
        point_phis = self._prepare_point_phi(n_points, first_point,
                                             n_by_layer, phi0_by_layer,
                                             dphi_by_layer)
        point_x, point_y = self._prepare_point_cartesian(point_rhos, point_phis)
        return point_x, point_y, point_layers

    def _get_first_point(self, n_by_layer):
        """
        Returns the point_id of the first point in each layer

        :return: numpy array of first point in each layer
        """
        first_point = np.zeros(len(n_by_layer), dtype=int)
        for i in range(len(n_by_layer)):
            first_point[i] = sum(n_by_layer[:i])
        return first_point

    def _prepare_points_lookup(self, n_by_layer):
        """
        Prepares lookup table to map from [layer_id, point_index] -> point_id

        :return:
        """
        lookup = np.zeros([len(n_by_layer),
                           max(n_by_layer)], dtype='int')
        lookup[:, :] = - 1
        point_id = 0
        for layer_id, layer_size in enumerate(n_by_layer):
            for cell_id in range(layer_size):
                lookup[layer_id, cell_id] = point_id
                point_id += 1
        assert point_id == sum(n_by_layer)
        return lookup

    def _prepare_point_rho(self, point_0, n_points, n_by_layer, r_by_layer):
        """
        Prepares lookup table to map from point_id to the radial position

        :return: numpy.array of shape [n_points]
        """
        radii = np.zeros(n_points, dtype=float)
        for lay, size in enumerate(n_by_layer):
            radii[point_0[lay]:point_0[lay] + size] = r_by_layer[lay]
        return radii

    def _prepare_point_phi(self, n_points, point_0, n_by_layer,
                                 phi0_by_layer, dphi_by_layer):
        """
        Prepares lookup table to map from point_id to the angular position

        :return: numpy.array of shape [n_points]
        """
        angles = np.zeros(n_points, dtype=float)
        for lay, layer_size in enumerate(n_by_layer):
            for point in range(layer_size):
                angles[point_0[lay] + point] = (phi0_by_layer[lay]
                                              + dphi_by_layer[lay]*point)
        angles %= 2 * math.pi
        return angles

    def _prepare_point_cartesian(self, point_rhos, point_phis):
        """
        Returns the positions of each point in cartesian system

        :return: pair of numpy.arrays of shape [n_points],
         - first one contains x coordinates
         - second one contains y coordinates
        """
        x_coor = point_rhos * np.cos(point_phis)
        y_coor = point_rhos * np.sin(point_phis)
        return x_coor, y_coor

    def _prepare_point_distances(self):
        """
        Returns a numpy array of distances between points

        :return: numpy array of shape [n_points,n_points]
        """
        point_xy = np.column_stack((self.point_x, self.point_y))
        distances = pdist(point_xy)
        return squareform(distances)

    def _prepare_point_neighbours(self, phi0_by_layer):
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
                    a_point = rel_pos - (phi0_by_layer[a_lay]/(2*math.pi))
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

    def _prepare_dphi_by_layer(self, n_by_layer):
        """
        Returns the phi separation of the points as defined by the number of
        points in the layer
        """
        return 2 * math.pi / np.asarray(n_by_layer)

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

    def shift_wires(self, shift_size, point_id=None):
        """
        Get the index of the wire that is displaced from point_id by
        shift_size points counter clockwise in the same layer,
        respecting periodicity.

        :return: index of point shift_size  counter clockwise of point_id
        """
        if point_id is None:
            point_id = np.arange(self.n_points)
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


class CDC(CylindricalArray):
    def __init__(self, projection=0.5):
        """
        Defines the Cylindrical Detector Geometry
        """
        # Number of wires in each layer
        cdc_wires = [198, 204, 210, 216, 222, 228, 234, 240, 246,
                     252, 258, 264, 270, 276, 282, 288, 294, 300]
        # Radius at end plate
        cdc_radii = [53.0, 54.6, 56.2, 57.8, 59.4, 61.0, 62.6, 64.2, 65.8,
                     67.4, 69.0, 70.6, 72.2, 73.8, 75.4, 77.0, 78.6, 80.2]
        # Phi0 at end plate
        # cdc_phi0 = [0.015867, 0.015400,
                     # 0.000000, 0.014544,
                     # 0.00000, 0.000000,
                     # 0.013426, 0.000000,
                     # 0.012771, 0.00000,
                     # 0.012177, 0.000000,
                     # 0.011636, 0.000000,
                     # 0.00000, 0.000000,
                     # 0.010686, 0.000000]
        # Phi0 in the middle plane
        cdc_phi0 = [-0.079333, 0.107800,
                    -0.089760, 0.101810,
                    -0.084908, 0.082674,
                    -0.067127, 0.078540,
                    -0.063853, 0.074800,
                    -0.060884, 0.071400,
                    -0.058177, 0.068296,
                    -0.077983, 0.076358,
                    -0.064114, 0.073304]
        # Add needed  180 degree shift for wire ID to match with global to local
        # coordinate shift
        cdc_phi0 = [phi_0 + np.pi for phi_0 in cdc_phi0]
        # Define the maximum angular shift of the wires in each layer from end
        # plate to the next
        self.phi_shft = np.array([-0.190400, 0.184800, -0.179520, 0.174533,
                                  -0.169816, 0.165347, -0.161107, 0.157080,
                                  -0.153248, 0.149600, -0.146121, 0.142800,
                                  -0.139626, 0.136591, -0.155966, 0.152716,
                                  -0.149600, 0.146608])

        cdc_phi0 = [p0 - ps/2. for p0, ps in zip(cdc_phi0, self.phi_shft)]
        dphi_from_phi0 = self.theta_at_rel_z(projection)
        new_radius = self.radius_at_theta(cdc_radii, dphi_from_phi0)
        new_dphi = dphi_from_phi0 + cdc_phi0

        # Build the cylindrical array
        point_x, point_y, layer_id = self._old_constructor(cdc_wires,
                                                           new_radius,
                                                           new_dphi)
        #CylindricalArray.__init__(self, cdc_wires, new_radius, new_dphi)
        # Build the cylindrical array
        CylindricalArray.__init__(self, point_x, point_y, layer_id)

        # Give it a recbe wiring
        self.recbe = RECBE(self)

    def theta_at_rel_z(self, z_dist, total_z=1.0):
        """
        Get the angular displacement of the wires in each layer as a function of
        the relative z_distance traversed

        :param z_dist:   z distance down CDC volume
        :param total_z:  total z distance of CDC volume

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
        return abs(radius*np.cos(self.phi_shft/2.)/\
                          np.cos(self.phi_shft/2. - this_theta))

class CTH(CylindricalArray):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, left_handed=True):
        """
        Defines the CTH geometry.  Note Cherenkov counter and light guide read
        out to the same volume here
        """
        self.left_handed = left_handed
        self.n_crystals = 64
        # The first two rows are for the active volumes, third is for excluded
        # volumes
        cth_n_vols = [self.n_crystals, self.n_crystals,
                      self.n_crystals, self.n_crystals, self.n_crystals]
        cth_radii = [44.78, 48.28, 44.78, 48.28, 0]
        cth_phi0 = [(-360/(2.*self.n_crystals)) * np.pi/180.,
                    (-360/(2.*self.n_crystals)) * np.pi/180.,
                    (-180 + 360/(2.*self.n_crystals)) * np.pi/180.,
                    (-180 + 360/(2.*self.n_crystals)) * np.pi/180.,
                    0]
        point_x, point_y, layer_id = self._old_constructor(cth_n_vols,
                                                           cth_radii,
                                                           cth_phi0)
        CylindricalArray.__init__(self, point_x, point_y, layer_id)

        # Get drawing parameters
        ## WIDTH HEIGHT DEFLECTION_ANGLE
        self.cherenkov_params = [1, 9, 20]
        self.scintillator_params = [0.5, 9, 13]
        # Convenience naming
        self.cher_crys = self.point_lookup[0:4:2].flatten()
        self.scin_crys = self.point_lookup[1:4:2].flatten()
        self.up_crys = self.point_lookup[0:2].flatten()
        self.up_cher = self.point_lookup[0]
        self.up_scin = self.point_lookup[1]
        self.down_crys = self.point_lookup[2:4].flatten()
        self.down_cher = self.point_lookup[2]
        self.down_scin = self.point_lookup[3]
        self.fiducial_crys = self.point_lookup[:4].flatten()

    def _get_channel_bits(self, channel):
        """
        Get the integer representation of the 10 most significant bits of the
        channel id
        """
        return int("{0:025b}".format(channel)[:10], 2)

    def chan_to_row(self, channel):
        """
        Maps :
            Upstream:
                Cherenkov counters and LG : 0
                Scintillators             : 1
                Scintillator LG           : 4
            Downstream:
                Cherenkov counters and LG : 2
                Scintillators             : 3
                Scintillator LG           : 4
        """
        # Mask out the CTH channel bits
        trimmed_channel = self._get_channel_bits(channel)
        # MSB is upstream mask
        is_upstream = 1 << 9
        # LSB is light guide boolean mask
        is_lg_mask = 1 << 0
        # Next bit is scintillator mask
        is_sc_mask = 1 << 1
        # Map downstream to rows [3-5]
        row_offset = 0
        if not trimmed_channel & is_upstream:
            row_offset = 2
        # Map both cherenkov light guide and cherenkov counter to same volume
        if not trimmed_channel & is_sc_mask:
            return 0 + row_offset
        # Map the scintillator to the next row
        elif not trimmed_channel & is_lg_mask:
            return 1 + row_offset
        # Ignore the scintillator light guide
        return 4

    def chan_to_module(self, channel):
        """
        Return upstream or downstream module flag from channel ID
        """
        # Mask out the CTH channel bits
        trimmed_channel = self._get_channel_bits(channel)
        # LSB is light guide boolean mask
        is_upstream = 1 << 9
        return bool(trimmed_channel & is_upstream)


    def _prepare_dphi_by_layer(self, n_by_layer):
        """
        Returns the phi separation of the points as defined by the number of
        points in the layer
        """
        if self.left_handed:
            # Downstream is left handed, upstream is right handed
            handedness = np.asarray([-1, -1, 1, 1, 1])
            return 2 * math.pi / (np.asarray(n_by_layer) * handedness)
        return 2 * math.pi / np.asarray(n_by_layer)

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
        point_x, point_y, layer_id = self._old_constructor(n_track_cent,
                                                           r_track_cent,
                                                           phi0_track_cent)
        CylindricalArray.__init__(self, point_x, point_y, layer_id)

class RECBE(CylindricalArray):
    def __init__(self, cdc, file_name=None):
        # Default file name
        recbe_file = "~/development/ICEDUST/"+\
                     "track-finding-yandex/data/chanmap_20160814.root"
        # Set file name
        if not file_name is None:
            recbe_file = file_name
        # Get the sense wires
        selection = "isSenseWire == 1 && LayerID > 0 && LayerID < 19"
        recbe_arr = root2array(recbe_file, selection=selection,
                               branches=["LayerID", "CellID", "BoardID",
                                         "BrdLayID", "BrdLocID", "ChanID"])
        recbe_arr["LayerID"] = recbe_arr["LayerID"] - 1
        # Get the board mapping
        self.wire_to_board = recbe_arr["BoardID"][cdc.point_lookup[\
                                                    recbe_arr["LayerID"],
                                                    recbe_arr["CellID"]]]
        brds = np.unique(self.wire_to_board)
        self.board_to_wires = \
            np.array([np.where(self.wire_to_board == val)[0] for val in brds])

        # Get the positions of the board
        board_x = np.array([np.average(cdc.point_x[self.board_to_wires[brd]])\
                                                           for brd in brds])
        board_y = np.array([np.average(cdc.point_y[self.board_to_wires[brd]])\
                                                           for brd in brds])

        # Get the board layers by board ID
        all_ids = np.unique(recbe_arr["BoardID"])
        board_lay = np.zeros(len(all_ids))
        for brd in all_ids:
            brd_id_wires = np.where(recbe_arr["BoardID"] == brd)
            board_lay[brd] = np.unique(recbe_arr["BrdLayID"][brd_id_wires])[0]

        # Construct the array
        CylindricalArray.__init__(self, board_x, board_y, board_lay)
