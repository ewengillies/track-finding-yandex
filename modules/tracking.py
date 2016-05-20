import numpy as np
import math
import sys
from scipy.sparse import lil_matrix, find, block_diag, hstack,\
        csr_matrix, identity
from scipy.spatial.distance import cdist
from cylinder import TrackCenters

"""
Notation used below:
 - wire_id is flat enumerator of all wires (from 0 to 4985)
 - layer_id is the index of layer (from 0 to 19)
 - cell_id is the index of wire in the layer (from 0 to layer_size -1)
"""


class HoughSpace(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=no-name-in-module
    def __init__(self, geom, sig_rho=33.6, sig_rho_max=35.,
                 sig_rho_min=24, sig_rho_sgma=3., trgt_rho=20., rho_bins=20,
                 arc_bins=20):
        """
        This class represents a Hough transform method. It initiates from a data
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
        :param rho_bins: Bins used in the radial direction
        :param arc_bins:  Number of angular bins in smallest track center ring
        """

        self.geom = geom
        self.sig_rho = sig_rho
        self.sig_rho_max = sig_rho_max
        self.sig_rho_min = sig_rho_min
        self.sig_rho_sgma = sig_rho_sgma
        self.trgt_rho = trgt_rho

        # Set the geometry of the TrackCenters to cover regions where the signal
        # track passes through the target and the CyDet volume.  Specifically,
        # enforce that the track's outer most hits may lie in the first or last
        # layer.
        r_max = self.geom.r_by_layer[-1] - self.sig_rho_max
        r_min = max(self.sig_rho_max - self.trgt_rho,
                    self.geom.r_by_layer[0] - self.sig_rho_max)
        self.track = TrackCenters(arc_bins=arc_bins, rho_bins=rho_bins, r_min=r_min, r_max=r_max)

        self.track_wire_dists = self._prepare_track_distances()
        self.correspondence = self._prepare_wire_track_corresp()
        self.norm_track_neighs = self._prepare_track_nns()

    def _prepare_track_distances(self):
        """
        Returns a numpy array of distances between tracks and wires

        :return: numpy array of shape [n_wires,n_tracks]
        """
        wire_xy = np.column_stack((self.geom.point_x,
                                   self.geom.point_y))
        trck_xy = np.column_stack((self.track.point_x, self.track.point_y))
        distances = cdist(wire_xy, trck_xy)
        return distances

    def dist_prob(self, distance):
        """
        Defines the probability distribution used for correspondence matrix

        :return: Gaussian of distance
        """
        distance -= self.sig_rho
        # Lower radii return a fitted gaussian function
        if distance < 0:
            return math.exp(-(distance**2)/(2.*(self.sig_rho_sgma**2))) + 0.05
        # Higher radii retun a linear decrease to just over the max value
        if distance >= 0:
            return 1.05 - distance/(self.sig_rho_max - self.sig_rho + 0.1)

    def _prepare_wire_track_corresp(self):
        """
        Defines the probability that a given wire belongs to a track centered at
        a given track center bin.  Produces two appended sparce matricies, one
        for even one for odd.

        :returns: scipy.sparse matrix of shape [n_wires, n_track_bin]
        """
        corsp = lil_matrix((self.geom.n_points, self.track.n_points))
        # Loop over all track centers
        for trck in range(self.track.n_points):
            # Loop over all wires in CyDet
            for wire in range(self.geom.n_points):
                # Calculate how far the wire is from the signal track centered
                # at the current track center
                this_dist = self.track_wire_dists[wire, trck]
                # If its within tolerance, return a probability
                if (this_dist <= self.sig_rho_max) and  \
                   (this_dist >= self.sig_rho_min):
                    corsp[wire, trck] = self.dist_prob(this_dist)
        # Define even and odd layer wires
        even_wires = self.geom.point_pol != 1
        odd_wires = self.geom.point_pol == 1
        # Make even and odd layer hough matricies
        corsp_odd = corsp.copy()
        corsp_odd[even_wires, :] = 0
        corsp_even = corsp.copy()
        corsp_even[odd_wires, :] = 0
        # Stack all submatricies horizontally
        corsp = hstack([corsp_even, corsp_odd])
        return corsp

    def _prepare_track_nns(self):
        """
        Normalized nearest neighbors matrix for tracks, where each cell is also
        its own neighbour

        :return: Sparse array of 3x5 neighbour relations
        """
        ## Pick out the neighbours
        nns = self.track.point_neighbours
        ## Add a track centre as its own neighbour
        nns = nns + identity(nns.shape[1])
        ## Extend the neighbours out one to the left and one to the right
        ## Now a 3x5 block
        nns = nns.dot(self.track.lr_neighbours)
        ## Weight the closer neighbours as double the further ones
        nns += self.track.lr_neighbours
        ## Normalize
        nns = csr_matrix(nns / nns.sum(axis=1))
        return nns

    def get_track_correspondence(self, track_id, values=False):
        """
        Returns the indices and values of the wires with non-zero
        correspondence to track center labeled by track_id

        :param values: returns the values as the second return value
        :return: indices of the wires with non-zero correspondence, optionally
                 returns corresponding value
        """
        corr_both = find(self.correspondence[:, track_id])
        corr_wire = corr_both[0]
        corr_value = corr_both[2]
        if values:
            return corr_wire, corr_value
        else:
            return corr_wire

    def get_wire_correspondence(self, wire_id, values=False):
        """
        Returns the indices and values of the track centers with non-zero
        correspondence to wire labeled by track_id

        :param values: if true, returns the correspondence values as the second

                       return value
        :return: indices of the wires with non-zero correspondence, optionally
                 returns corresponding value
        """
        corr_both = find(self.correspondence[wire_id, :])
        corr_track = corr_both[1]
        corr_value = corr_both[2]
        if values:
            return corr_track, corr_value
        else:
            return corr_track

class HoughTransformer(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=no-name-in-module
    def __init__(self, hough_matrix, track_neighs, fit_wires,
                 min_percentile=0.97, regular=50., alpha_rw=2, alpha_max=2):
        """
        A class that handles the specific normalizations and reweighting of the
        hough image.

        :param hough_matrix:    Hough transform matrix
        :param track_neighs:    Nearest neighbour matrix for tracks
        :param min_percentile:  Bottom percentile ignored in hough space
        :param alpha_rw:        Weight of reweight in hough space
        :param alpha_max:       Weight of peak reweight in hough space
        :param regular:         Regularization used to normalize correspondence
                                matrix
        """
        self.hough_matrix = hough_matrix
        self.min_percentile = min_percentile
        self.alpha_rw = alpha_rw
        self.alpha_max = alpha_max
        # Normalize the hough transform so to decrease bias towards track-centre
        # layers  with better coverage, i.e. more wires in range
        self.normed_corresp = \
            csr_matrix(hough_matrix/(hough_matrix.sum(axis=1)+regular))
        self.track_neighs = track_neighs
        # Fit the distribution information to a given hit wire weighted output
        self.wire_mean, self.image_mean, self.percs, self.perc_values =\
                None, None, None, None
        self.fit(fit_wires)

    def is_max(self, image, alpha=10):
        """
        This is soft version of 'is_max' rule.
        Strong version returns 1 it it is max among neighbors, 0 otherwise.
        The greater alpha, the closer we are to 'strong' version

        :param image:  Hough image(s) from transformation(s)
        :param alpha:  Weight of exponential reweighting
        """
        # Exponentially reweight the data
        exponents = np.exp(alpha * image)
        # Check the number of maxima we expect to get back
        # Note, we expect two maxima for each tested hough transform
        # One for even, one for odd.
        # Testing multiple transforms scales as 2*n_transforms
        n_parts = image.shape[1] // self.track_neighs.shape[1]
        assert n_parts * self.track_neighs.shape[1] == image.shape[1]
        # Block diagnol matrix, with each block being one copy of the
        # neighs_matrix
        full_neigh = block_diag([self.track_neighs]*n_parts, format='csr')
        # Return the value at the point
        # normalized the sum of its values and its neighbouring values
        return exponents / full_neigh.dot(exponents.T).T

    def fit(self, fit_wires):
        """
        Fit the hough transform to the weighted hit data on the wire array

        :param fit_wires: Data to be fit
        """
        # Center the input distribution around 0
        self.wire_mean = fit_wires.mean()
        original = fit_wires - self.wire_mean
        # Transform into hough space
        hough_images = self.normed_corresp.T.dot(original.T).T
        # Use a percentile binning with increased sampling near 1
        perc = np.linspace(0, 1, 200) ** 0.5
        # Get the percentile distribution
        self.percs = np.percentile(hough_images.flatten(), perc * 100)
        # Remove the bottom min_percentile
        self.perc_values = np.maximum(0., perc - self.min_percentile)
        # Shift the remaining values to the [0-1] range
        self.perc_values /= (1. - self.min_percentile)
        hough_images = np.interp(hough_images,\
                self.percs, self.perc_values)
        # Sharpen locally maximum peaks
        hough_images *= self.is_max(hough_images, alpha=self.alpha_max)
        # Exponentiate the image
        hough_images = np.exp(self.alpha_rw * hough_images)
        self.image_mean = hough_images.mean()
        return self

    def transform(self, trans_wires, only_hits=True, flatten=True):
        """
        Transform the data according to the fit

        :param X: Data to be transformed

        :return after_hough, hough_images: ???
        """
        # Center the input distribution around 0
        original = trans_wires - self.wire_mean
        # Record the wires with hits if need be
        # TODO clean this up, i.e. check it always works etc
        if only_hits:
            hit_wires = np.nonzero(trans_wires)
        # Perform the hough transform
        hough_images = self.normed_corresp.T.dot(original.T).T
        # Remove the bottom min_percentile shift the remaining range to [0-1]
        hough_images = np.interp(hough_images,\
                self.percs, self.perc_values)
        # Reweight hough images by how locally maximal each point is
        hough_images *= self.is_max(hough_images, alpha=self.alpha_max)
        # Reweight result exponentially
        hough_images = np.exp(self.alpha_rw * hough_images)
        # Center around 0
        hough_images -= self.image_mean
        # Inverse hough transform
        after_hough = self.normed_corresp.dot(hough_images.T).T
        if only_hits:
            after_hough = after_hough[hit_wires]
        if flatten:
            after_hough = after_hough.flatten()
        return after_hough, hough_images

class HoughShifter(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=bad-continuation
    # pylint: disable=no-name-in-module
    def __init__(self, hough, upper_lim, lower_lim, dphi=None):
        """
        A class that handles shifting the even layers to remove the stereometric
        effects

        :param hough:      Hough instance used to deduce geometry of track
                           centres
        :param dphi:       Unit of angular shift used when shifting aligning
                           hough image
        :param upper_lim:  Upper limit when scanning dphi, in units of dphi
        :param lower_lim:  Lower limit when scanning dphi, in units of dphi
        """
        self.hough = hough
        self.geom = hough.geom
        self.upper_lim = upper_lim
        self.lower_lim = lower_lim
        # Default for dphi is the angular separation of the points in the inner
        # most tracking layer
        if dphi == None:
            self.dphi = 2*np.pi/self.hough.track.n_by_layer[0]
        else:
            self.dphi = dphi
        self.phi_slices = self._get_slices()
        self.shifters = self._get_shifters()
        # Initialize the indecies used to shift the image to the ideal angle
        self.rotate_index = None

    def _get_slices(self):
        """
        Returns a sparse matrix that denotes which hough track centers belong to
        each slice in phi.  The shape is:

        (number of slices in phi) x (number of track centres)

        If a track j is in slice i, then phi_slices[i][j] is non-zero.  It is
        zero otherwise.  Each row of slices is normalized by the number of track
        centres it has.

        :return:        CSR matrix defining slices in phi and the corresponding
                        track centre points
        """
        # Determine the number of slices in phi
        n_phis = np.ceil(2*np.pi/self.dphi).astype(int)
        # Initialize the set of points in the previous slice
        prev_slice = np.zeros(0)
        # Initialize the returned sparse matrix
        all_slices = lil_matrix((n_phis, self.hough.track.n_points))
        for phi_slice in range(n_phis):
            # Define phi range for this slice
            phi_0 = phi_slice*self.dphi
            phi_1 = phi_0 + self.dphi
            # Get the track centres in this range
            this_slice = np.where(\
                    (self.hough.track.point_phis >= phi_0) &\
                    (self.hough.track.point_phis < phi_1))[0]
            # Avoid double counting a track centre
            this_slice = np.setdiff1d(this_slice, prev_slice)
            prev_slice = this_slice
            # Add these points to the slice
            all_slices[phi_slice, this_slice] = 1.
        # Normalize by the number in each slice to get density in phi
        return csr_matrix(all_slices / all_slices.sum(axis=1))

    def _shift_wire_ids(self, wire_shift):
        """
        Shift each wire ID by the amount of wire centres given.  Note that this
        calls the cydet.shift_wire function, so wires locations themselves are
        unchanged, but each wire ID is mapped to a new wire location.

        :param wire_shift:  The amount each wire is shifted, where positive
                            denotes the counter clockwise direction
        :return:            The new ID for each wire
        """
        return np.array([self.geom.shift_wire(wire, shift)\
                            for wire, shift\
                            in zip(np.arange(self.geom.n_points), wire_shift)])

    def _get_even_shift(self, phi_shift=2*np.pi/81.):
        #TODO try adding option to shift odd layers as well
        """
        Shift the even layers by phi_shift to return a new wire index after the
        shift.

        :param phi_shift:   The angular shift of the even layer

        :return forward_index:   The new index of each wire
        :return backward_index:  The old index of each wire after it has been
                                 shifted
        """
        # Initialize the shift by layer
        shift_by_layer = np.zeros(len(self.geom.dphi_by_layer))
        # Shift the even layers clockwise by the closest number of wire cells to
        # the desired phi_shift
        shift_by_layer[::2] = -np.round(phi_shift/self.geom.dphi_by_layer)[::2]
        shift_by_layer = shift_by_layer.astype(int)
        # Denote this shift by wire ID
        shift_by_wire = np.take(shift_by_layer, self.geom.point_layers)
        # Determine the forward shift to the aligned hough space
        forward_index = self._shift_wire_ids(shift_by_wire)
        # Record the backward shift by inverting the mapping
        backward_index = np.argsort(forward_index)
        # Return both the forward and backward shifts
        return forward_index, backward_index

    def _get_shifters(self):
        """
        Get the shifted wire IDs for a the allowed range of dphis.  Note that
        the zeroth element of the returned shifters corresponds to the lower
        limit of the shift in phi, whereas the last element is the upper limit
        of the shift.

        :return forward_shift:  Forward shifted wire IDs for each allowed dphi.
                                Shape is [n_dphi, n_wires]
        :return backward_shift: Backward shifted wire IDs for each allowed dphi.
                                Shape is [n_dphi, n_wires]
        """
        # Initialize all the desired range of dphis to test
        phi_range = np.arange(self.lower_lim, self.upper_lim+1)*self.dphi
        # Collect all the shifted IDs as [forward/backward, n_wires, n_dphis]
        forward_shift, backward_shift = np.dstack(\
                np.vstack(self._get_even_shift(phi_shift))\
                for phi_shift in phi_range)
        # Return the needed shape
        return forward_shift.T, backward_shift.T

    def _get_ideal_shift(self, even_slices, odd_slices):
        """
        Given an integration in hough space due to the even and odd layer
        contributions, determine the ideal shift in phi such that these
        contributions maximally overlap.

        :param even_slices:  contribution to hough space slices due to even
                             layer hits
        :param odd_slices:   contribution to hough space slices due to odd
                             layer hits
        :return ideal_phi:   Angular shift to maximally overlap these two in
                             units of dphi
        """
        # Find the ideal shift in phi to align images in hough space
        diff = sys.maxint
        # Try shifting the integrated contributions by each of the allowed dphis
        for phi_shift in range(self.lower_lim, self.upper_lim+1):
            # Shift the integral of contributions over by a given dphi
            new_even_slices = np.roll(even_slices, phi_shift)
            # Get the sum of the square of the difference
            this_diff = np.sum(np.square(new_even_slices - odd_slices))
            # Check for the best overlap
            if this_diff < diff:
                diff = this_diff
                ideal_phi = phi_shift
        # Return the phi denoting the best overlap
        return ideal_phi

    def fit_shift(self, hough_image_even, hough_image_odd):
        """
        Find the ideal rotations for each pair of even and odd contributions to
        hough space

        :param hough_image_even:  The hough space contributions from the even
                                  layers.  Shape is [n_images x n_track_centres]
        :param hough_image_odd:   The hough space contributions from the odd
                                  layers.  Shape is [n_images x n_track_centres]

        :return ideal_roate:      Ideal rotation for each image in radians
        :return slices_even:      Integral of hough space before shift due to
                                  even layers
        :return slices_odd:       Integral of hough space before shift due to
                                  odd layers
        """
        # Record the number of images
        assert hough_image_even.shape[0] == hough_image_odd.shape[0],\
            'Number of even images, {} '.format(hough_image_even.shape[0])+\
            'does not match the '+\
            'number of odd images, {}'.format(hough_image_odd.shape[0])
        n_images = hough_image_even.shape[0]
        # Get even and odd contributions
        slices_even = self.phi_slices.dot(hough_image_even.T).T
        slices_odd = self.phi_slices.dot(hough_image_odd.T).T
        # Find the ideal rotations
        ideal_rotate = np.array(\
                [self._get_ideal_shift(slices_even[evt, :], slices_odd[evt, :])\
                                 for evt in range(n_images)])
        # Store the indexes to reference the correct shifters later
        self.rotate_index = ideal_rotate - self.lower_lim
        # Return a physical angle
        ideal_rotate = ideal_rotate*self.dphi
        return ideal_rotate, slices_even, slices_odd

    def shift_result(self, results, backward=False):
        """
        Rotate the results for a set of events to the ideal rotations, as
        determined in fit_shift().

        :param results: Event wise results to be shifted.  The shape should be
                        [n_evts x n_wires]
        """
        # Check this instance has been fit to some hough images first
        assert self.rotate_index is not None,\
                "The shifter has not been fit to a set of hough images yet. "+\
                "Call HoughShifter.fit_shift() first."
        # Check that this instance has been fit to the hough image corresponding
        # to the results we are now shifting
        assert results.shape[0] == self.rotate_index.shape[0],\
                "The number of ideal rotations and the number of events in "+\
                "the result do not match.  Check this instance has been fit "+\
                "to the hough image from these results."
        # Select the direction we will be shifting the result
        if backward == True:
            this_shift = self.shifters[1]
        else:
            this_shift = self.shifters[0]
        # Returned a rotated image of the result
        return np.vstack(evt_res[this_shift[evt_rot]]\
                for evt_res, evt_rot in zip(results, self.rotate_index))
