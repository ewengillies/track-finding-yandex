 # Plotting Methods for All Analyses

#get_ipython().magic('pylab inline')
from collections import OrderedDict
import numpy as np
import pandas
from scipy.spatial import ConvexHull
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

DOT_SIZE = 40

# Declare Globally
N_WIRES = 4482
PLOT_HOUGH = None

def plot_set_font(size=15):
    """
    Default Font for all plots
    """
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : size}
    # Enable LaTeX support in fonts
    matplotlib.rc('font', **font)

def plot_set_hough(hough_instance):
    """
    Setup track centre locations in x, y and r, theta
    """
    PLOT_HOUGH = hough_instance

def plot_setup_for_detector(size=(15, 15), no_labels=False):
    """
    Default plot for event display
    """
    plot_set_font()
    fig = plt.figure(1, figsize=size)
    axis = fig.add_subplot(111, projection='polar')
    axis.set_ylim([0, 85])
    # Set theta ticks at 45 degrees
    thetaticks = np.arange(0, 360, 45)
    # Set ticks further out from edge
    axis.set_thetagrids(thetaticks, frac=1.05)
    # Set radial ticks
    axis.set_yticks(np.arange(10, 81, 20))
    if no_labels:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    return axis, fig

def plot_get_hits(labels):
    """
    Gets the signal and background hits for an event

    :param labels: Labels from an event

    :return: signal hits, background hits
    """
    # Ensure the event is the right number of wires
    assert len(labels) == N_WIRES, \
        "Number of input wires is {}, real number is {}".format(len(labels),
                                                                N_WIRES)
    # Get the signal hits
    sig_hits = np.where(labels == 1)
    bkg_hits = np.where(labels == 2)
    return sig_hits, bkg_hits

def plot_output(labels, cydet, size=DOT_SIZE*np.ones(N_WIRES), tolerance=0.,
                sig_color="blue", bkg_color="red", figsize=(12, 14),
                add_transform=False, hough=PLOT_HOUGH,
                rotate_even=0, no_labels=False, recbe=None, **kwargs):
    """
    Draw the output of a classifier by scaling the hits to the size of the
    output.

    :param labels:    Triple labels to tell signal from background from empty
    :param cydet:     CyDet to be used for geometry of wires
    :param size:      Size of all hits on every wire, this includes empty wires
    :param sig_color: Colour of signal hits, default blue
    :param bkg_color: Colour of background hits, default red

    :return: axis of plot, figure of plot
    """
    # Set the default font
    plot_set_font()
    # Get the hits from the labels
    sig_hits, bkg_hits = plot_get_hits(labels)
    # Get axis and figure
    fig = plt.figure(1, figsize=figsize)
    axis = fig.add_subplot(111, projection='polar')
    axis.set_ylim([0, 85])
    # Set theta ticks at 45 degrees
    thetaticks = np.arange(0, 360, 45)
    # Set ticks further out from edge
    axis.set_thetagrids(thetaticks, frac=1.05)
    # Set radial ticks
    axis.set_yticks(np.arange(10, 81, 20))
    if no_labels:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    # Plot all wires
    wire_rhos = cydet.get_points_rhos_and_phis()[0].copy()
    wire_phis = cydet.get_points_rhos_and_phis()[1].copy()
    #Add a rotation to the layers
    if rotate_even != 0:
        even_wires = np.where(cydet.point_pol == 0)[0]
        wire_phis[even_wires] += rotate_even
    axis.scatter(wire_phis, wire_rhos, marker='.', s=5, alpha=0.2)
    # Plot signal hits
    axis.scatter(wire_phis[sig_hits], wire_rhos[sig_hits], s=size[sig_hits],
                 marker='o', color=sig_color, **kwargs)
    # Plot background hits
    axis.scatter(wire_phis[bkg_hits], wire_rhos[bkg_hits], s=size[bkg_hits],
                 marker='o', color=bkg_color, **kwargs)
    # Add the transform if its wanted
    if add_transform:
        shading = plot_get_transform_shading(hough)
        draw_trans = np.where(size > np.amax(size)*tolerance)[0]
        trans_w = plot_norm_size(size, 1.)
        for wire in draw_trans:
            plot_add_transform(wire, cydet, hough,
                               inverse=False, prob_spread=shading,
                               l_alpha=trans_w[wire],
                               s_alpha=trans_w[wire]/len(shading))
    # Add the boards if they are wanted
    if not recbe is None:
        plot_add_recbe(labels, cydet, size,
                       sig_color=sig_color, bkg_color=bkg_color, **kwargs)

    # Return axis and figure
    return axis, fig

def plot_add_recbe(labels, cydet, size=np.ones(N_WIRES),
                   recbe_labels=None, recbe=None, recbe_alpha=None,
                   add_transform=False, hough=PLOT_HOUGH, tolerance=0.0,
                   sig_color="blue", bkg_color="red", **kwargs):
    # Get the geometry we are using
    use_recbe = cydet.recbe
    if not recbe is None:
        use_recbe = recbe
    # Get board scores
    if recbe_alpha is None:
        recbe_alpha = np.zeros(use_recbe.board_to_wires.shape[0], dtype=float)
        for wire_id, wire_score in enumerate(size):
            recbe_alpha[use_recbe.wire_to_board[wire_id]] += wire_score
    # Get the labels
    sig_hits, bkg_hits = plot_get_hits(labels)
    if recbe_labels is None:
        # Get the board labels as "has signal hit"
        recbe_labels = np.zeros_like(recbe_alpha)
        recbe_labels[use_recbe.wire_to_board[bkg_hits]] = 2
        recbe_labels[use_recbe.wire_to_board[sig_hits]] = 1
    # Get where the labels are non-zero
    for board_id in np.where(recbe_labels == 1)[0]:
        plot_add_board(cydet, board_id, recbe_alpha[board_id],
                       color=sig_color, **kwargs)
    for board_id in np.where(recbe_labels == 2)[0]:
        plot_add_board(cydet, board_id, recbe_alpha[board_id],
                       color=bkg_color, **kwargs)
    # Add the transform if its wanted
    if add_transform:
        shading = plot_get_transform_shading(hough)
        draw_trans = np.where(recbe_alpha > np.amax(recbe_alpha)*tolerance)[0]
        trans_w = plot_norm_size(recbe_alpha, 1.)
        for board in draw_trans:
            plot_add_transform(board, use_recbe, hough,
                               inverse=False,
                               prob_spread=shading,
                               l_alpha=trans_w[board],
                               s_alpha=trans_w[board]/len(shading))

def plot_add_board(cydet, board_id, alpha, **kwargs):
    board_point_ids = cydet.recbe.board_to_wires[board_id]
    points = np.vstack([cydet.point_x[board_point_ids],
                        cydet.point_y[board_point_ids]]).T
    c_hull = ConvexHull(points)
    plot_verts = np.append(c_hull.vertices, c_hull.vertices[0])
    plot_brd = plt.Polygon(points[plot_verts],
                           True,
                           transform=plt.gca().transData._b,
                           alpha=alpha,
                           **kwargs)
    plt.gca().add_artist(plot_brd)

def plot_add_outlines(labels, cydet,
                      sig_color="blue", bkg_color="red",
                      size=DOT_SIZE, lw=0.5):
    """
    Adds in outlines of original locations of hits

    :param labels:    Triple labels to tell signal from background from empty
    :param cydet:     CyDet to be used for geometry of wires
    :param size:      Size of all hits on every wire, this includes empty wires
    :param sig_color: Colour of signal hits, default blue
    :param bkg_color: Colour of background hits, default red
    """
    # Get the hits from the labels
    sig_hits, bkg_hits = plot_get_hits(labels)
    # Add the outlines of the signal and background
    wire_rhos, wire_phis = cydet.get_points_rhos_and_phis()
    plt.gca().scatter(wire_phis[sig_hits], wire_rhos[sig_hits], s=size,
                      marker='o', edgecolors=sig_color, facecolors='none',
                      lw=lw)
    plt.gca().scatter(wire_phis[bkg_hits], wire_rhos[bkg_hits], s=size,
                      marker='o', edgecolors=bkg_color, facecolors='none',
                      lw=lw)

def plot_add_circle(x, y, radius, color="green",
                    lw=1., spread=0, l_alpha=1., s_alpha=0.025, fill=False,
                    **kwargs):
    """
    Add a circle to our plot

    :param x:        x location of circle centre
    :param y:        y location of circle centre
    :param radius:   radius of circle
    :param color:    color of circle
    :param lw:       line width of circle
    :param spread:   spread of circle, symmetric
    :param l_alpha:  overall normalization on weight of line
    :param s_alpha:  overall normalization on weight of spread
    """
    plot_circle = plt.Circle((x, y), radius, transform=plt.gca().transData._b,
                             color=color, fill=fill, alpha=l_alpha, lw=lw,
                             **kwargs)
    plt.gca().add_artist(plot_circle)
    # Add a spread if need be
    if spread != 0:
        plot_spread = plt.Circle((x, y), radius,
                                 transform=plt.gca().transData._b, color=color,
                                 alpha=s_alpha, fill=fill, lw=spread)
        plt.gca().add_artist(plot_spread)

def plot_get_transform_shading(hough):
    """
    Returns shading of visual hough transform

    :param hough: Hough to be used for geometry of transform

    :return: Shading of hough tranfrom
    """
    # Define the signal radius
    r_sig = hough.sig_rho
    r_max = hough.sig_rho_max
    r_min = hough.sig_rho_min
    print(r_sig, r_max, r_min)
    return np.array([hough.dist_prob(r)/hough.dist_prob(r_sig)
                     for r in  np.linspace(r_min, r_max, 10)])

def plot_add_transform(index, cydet, hough,
                       inverse=False, color="green",
                       l_alpha=1., s_alpha=0.025,
                       prob_spread=0.5*np.ones(10)):
    """
    Add a hough transform overlay, assumes forward transform unless told
    otherwise

    :param index:        index of wire or track centre
    :param cydet:        CyDet to be used for geometry of wires
    :param hough:        Hough to be used for geometry of transform
    :param inverse:      If true, plot from track centre to wires
    :param color:        color of circle
    :param l_alpha:      overall normalization on weight of line
    :param s_alpha:      overall normalization on weight of spread
    :param prob_spread:  Stored values of the correspondnce function, must be
                         length 10
    """
    # Define the signal radius
    r_sig = hough.sig_rho
    r_max = hough.sig_rho_max
    r_min = hough.sig_rho_min
    # Check if we are mapping forwards or inverse
    if inverse:
        plot_x, plot_y = hough.track.get_points_xs_and_ys()
    else:
        plot_x, plot_y = cydet.get_points_xs_and_ys()

    # Just plot this one point
    plot_x, plot_y = plot_x[index], plot_y[index]

    # Add the target radius circle
    plot_add_circle(plot_x, plot_y, r_sig,
                    color=color, l_alpha=l_alpha, s_alpha=s_alpha)
    # Add the spread around the target as 10 shaded circles at varying distances
    for n_rad, rad in enumerate(np.linspace(r_min, r_max, 10)):
        assert len(prob_spread) == 10, \
                "Length of prob_spread must be 10"
        # Add the spread
        plot_add_circle(plot_x, plot_y, rad,
                        color=color, l_alpha=s_alpha*prob_spread[n_rad], lw=3)

def plot_add_tracks(cydet, hough, tolerance=0.001, size=None,
                    color="DarkOrange", add_transform=False):
    """
    Add track centres as orange points

    :param tolerance:     Relative size of smallest drawn track centre to
                          maximum track centre
    :param size:          Size of all track centres
    :param color:         Colour of track centres
    :param add_transform: Adds default transform if true
    """
    # Get default size
    if size is None:
        size = (DOT_SIZE/5.)*np.ones(hough.track.n_points)
    # Find the tracks above tolerance of the max
    draw_tracks = np.where(size > np.amax(size)*tolerance)[0]
    print("Length of draw tracks are {}".format(len(draw_tracks)))
    # Draw these tracks
    track_rhos, track_phis = hough.track.get_points_rhos_and_phis()
    plt.gca().scatter(track_phis[draw_tracks], track_rhos[draw_tracks],
                      s=size[draw_tracks], marker='o', color=color, zorder=10)
    # Add the transform if its wanted
    # Add the transform if its wanted
    if add_transform:
        shading = plot_get_transform_shading(hough)
        trans_w = plot_norm_size(size, 1.)
        for trk in draw_tracks:
            plot_add_transform(trk, cydet, hough,
                               inverse=True,
                               prob_spread=shading,
                               l_alpha=trans_w[trk],
                               s_alpha=trans_w[trk]/len(shading))

def plot_norm_size(bdt_guess, size=DOT_SIZE):
    return (float(size)*bdt_guess)/np.amax(bdt_guess)

def plot_rocs(labels, predictions, names=None,
              zoom=False,
              zoom_x_lims=(70, 100),
              zoom_y_lims=(70, 100),
              fontsize=15,
              datum_label="",
              lw=1,
              weights=None,
              **kwargs):
    """
    Nice default plotting for roc curves
    """
    predicts = list(predictions.keys())
    if names is None:
        names = predicts
    #fig = plt.figure(1)
    #axis = [fig.add_subplot(111)]*len(names)

    if datum_label != "":
        datum_label += " "
    #for pred, ax, name in zip(predicts, axis, names):
    for pred in predicts:
        _, _ = plot_roc_curve(labels, predictions[pred][:, 1],
                              zoom=zoom,
                              zoom_x_lims=zoom_x_lims,
                              zoom_y_lims=zoom_y_lims,
                              fontsize=fontsize,
                              label=pred,
                              datum_label=datum_label,
                              lw=lw,
                              weights=weights,
                              **kwargs)

def plot_roc_curve(labels, predictions,
                   zoom=False,
                   fontsize=15,
                   zoom_x_lims=(70, 100),
                   zoom_y_lims=(70, 100),
                   label="", datum_label="",
                   weights=None,
                   **kwargs):
    """
    Nice default plotting for roc curves
    """
    # Get the figure and axis
    fig = plt.figure(1)
    axis = fig.add_subplot(111)
    # Get the FPR and TPR for the curve and scale up by 100
    fpr, tpr, _ = roc_curve(labels, predictions, sample_weight=weights)
    fpr *= 100.
    tpr *= 100.
    axis.xaxis.tick_top()
    axis.set_xlabel('Signal '+datum_label+' Retention Efficiency',
                    size=fontsize)
    axis.xaxis.set_label_position('top')
    axis.set_ylabel('Background '+datum_label+' Rejection Efficiency',
                    size=fontsize)
    axis.plot(tpr, 100-fpr, label=label, **kwargs)
    axis.grid(b=True, which='minor', color='grey', linestyle=':')
    axis.grid(b=True, which='major', color='grey', linestyle='--')

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    axis.xaxis.set_major_formatter(xticks)
    axis.yaxis.set_major_formatter(xticks)

    axis.set_axisbelow(True)
    if zoom:
        axis.set_xlim(zoom_x_lims)
        axis.set_ylim(zoom_y_lims)
    axis.minorticks_on()
    # Deal with the legend
    axis.legend(loc=3, frameon=1, fontsize=fontsize)
    leg = axis.get_legend()
    _ = leg.get_frame()
    return fig, axis

def plot_feature(sig_samp, bkg_samp, xlabel="", ylabel="", title="",
                 xlog=False, ylog=False, nbins=50, bkg_type=2, sig_type=1,
                 uniform_bins=True, normed=True, override_bins=None, tight=True,
                 tight_layout=False, **kwargs):
    """
    Nice default plotting for features
    """
    fig = plt.figure(1)
    axis = fig.add_subplot(111)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    # Apply logarithmic x
    if xlog:
        sig_samp = np.log10(sig_samp)
        bkg_samp = np.log10(bkg_samp)
        axis.set_xticks(np.arange(axis.get_xticks()[0], axis.get_xticks()[-1]))
        axis.set_xticklabels([str(10**power) for power in axis.get_xticks()])

    # Make the bins uniform
    if uniform_bins:
        nbins = plot_get_uniform_bins(sig_samp, bkg_samp, nbins)

    if override_bins is not None:
        nbins = override_bins

    # Plot the functions
    axis.hist(sig_samp, normed=normed, bins=nbins, label='Signal',
              log=ylog, alpha=0.5, color="blue", **kwargs)
    axis.hist(bkg_samp, normed=normed, bins=nbins, label='Background',
              log=ylog, alpha=0.5, color="red", **kwargs)

    leg = axis.legend(loc=0, frameon=1, fontsize=15)
    _ = leg.get_frame()
    if tight_layout:
        fig.tight_layout()
    return axis, fig

def plot_get_uniform_bins(sample_a, sample_b, nbins=20.):
    min_bin = np.amin(np.append(sample_a, sample_b))
    max_bin = np.amax(np.append(sample_a, sample_b))
    bins = np.arange(min_bin, max_bin, (max_bin - min_bin)/float(nbins))
    return bins

def plot_evt_feature(feature, labels,
                     xlabel="X Axis", ylabel="Y Axis", title="",
                     xlog=False, ylog=False, nbins=50):
    """
    Nice default plotting for features
    """
    sig = np.where(labels == 1)
    bkg = np.where(labels == 0)

    fig = plt.figure(1)
    axis = fig.add_subplot(111)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    if xlog:
        axis.hist(np.log10(feature[sig]), normed=True, bins=nbins,
                  label='Signal', log=ylog, alpha=0.5, color="blue")
        axis.hist(np.log10(feature[bkg]), normed=True, bins=nbins,
                  label='Background', log=ylog, alpha=0.5, color="red")
        axis.set_xticks(np.arange(axis.get_xticks()[0], axis.get_xticks()[-1]))
        axis.set_xticklabels([str(10**power) for power in axis.get_xticks()])
        #axis.set_xticklabels([r"$10^{"+str(power)+"}$" 
        #                     for power in axis.get_xticks()])
    else:
        axis.hist(feature[sig],
                  normed=True, bins=nbins, label='Signal',
                  log=ylog, alpha=0.5, color="blue")
        axis.hist(feature[bkg],
                  normed=True, bins=nbins, label='Background',
                  log=ylog, alpha=0.5, color="red")
    leg = axis.legend(loc=0, frameon=1, fontsize=15)
    _ = leg.get_frame()
    return axis, fig

def plot_feature_correlations(data_frame, title, labels_dict):
    fig = plt.figure()
    axis = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('RdBu', 30)
    cax = axis.imshow(data_frame.corr(), interpolation="nearest",
                      cmap=cmap, vmin=-1, vmax=1)
    plt.title(title)
    labels = [labels_dict[val] for val in data_frame.columns.values]
    labels = [None] + labels
    axis.set_xticklabels(labels, rotation=45)
    axis.set_yticklabels(labels)
    # Add colorbar, make sure to specify tick locations to match desired
    # ticklabels
    ticklabels = np.arange(-1, 1.001, 0.25)
    _ = fig.colorbar(cax, ticks=ticklabels)
    return axis, fig

def plot_hough_shift(int_even, int_odd, this_phi, savedir=None):
    dphi = (2.*np.pi)/len(int_even)
    phi_bins = dphi*np.arange(len(int_even))
    plt.bar(phi_bins, int_even, alpha=0.5, width=dphi)
    plt.bar(phi_bins, int_odd, color="red", alpha=0.5, width=dphi)
    plt.title(r"Track Center Weights vs. $\phi$")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"Track Center Weights (Integrated over $\rho$)")
    plt.xlim([0, 2*np.pi])
    if savedir != None:
        plt.savefig(savedir+"hough_by_phi.png")
    plt.show()
    plt.bar(phi_bins, int_even, alpha=0.5, width=dphi)
    plt.bar(np.roll(phi_bins, this_phi), int_odd, color="red", alpha=0.5,
            width=dphi)
    plt.title(r"Shifted Track Center Weights vs. $\phi$")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"Track Center Weights (Integrated over $\rho$)")
    plt.xlim([0, 2*np.pi])
    if savedir != None:
        plt.savefig(savedir+"s_hough_by_phi.png")
    plt.show()

def plot_add_cth_crystal(crys, geom, **kwargs):
    """
    Add a CTH crystal to an existing CDC plot

    :param crys: flat_id of the crystal to be added
    :param geom: copy of the CTH geometry to use
    """
    # Check what kind of crystal it is
    assert (crys >= 0 and crys < 256), \
        "Volume ID {} is out of range".format(crys)
    # Check which crystal
    if (crys < 64) or (crys >= 128 and crys < 192):
        these_params = geom.cherenkov_params
    elif (crys >= 64 and crys < 128) or (crys >= 192):
        these_params = geom.scintillator_params
    # Draw the crystal
    plot_draw_cth_crystal(crys, geom, these_params, **kwargs)

def plot_draw_cth_crystal(crys, geom, params, **kwargs):
    """
    Impliment actually drawing the crystal
    """
    # Get the parameters
    wth, hgt, ang = params
    angle = (geom.get_points_rhos_and_phis()[1][crys] * 180/np.pi) - ang
    a_plt = angle * np.pi/180
    # Rotate the plot correctly
    x_plt = geom.get_points_xs_and_ys()[0][crys] \
        + np.sin(a_plt)*hgt/2. - np.cos(a_plt)*wth/2.
    y_plt = geom.get_points_xs_and_ys()[1][crys] \
        - np.cos(a_plt)*hgt/2. - np.sin(a_plt)*wth/2.
    # Add the rectangle
    plot_rect = plt.Rectangle((x_plt, y_plt), wth, hgt, angle=angle,
                              transform=plt.gca().transData._b, **kwargs)
    plt.gca().add_artist(plot_rect)

def plot_add_cth_outlines(geom, **kwargs):
    """
    Add the outlines of all the crystals to the CDC plot
    """
    # Loop through and draw all crystals
    for crystal in range(256):
        plot_add_cth_crystal(crystal, geom,
                             fill=False, alpha=0.2, lw=0.4, **kwargs)

def plot_add_crystals(output, geom, **kwargs):
    """
    Plot the crystals in an arbitrary colour

    :param output: vector/list of hit volumes of shape [geom.n_points]
                   key: 0 is nothing, 1 is for plotting
    :param geom: CTH geometry to draw
    """
    for crystal in [i for i, e in enumerate(output) if e]:
        plot_add_cth_crystal(crystal, geom, lw=0.4, **kwargs)

def plot_add_cth(output, trig_out, geom, **kwargs):
    """
    Plot the output hit crystals

    :param output: vector/list of hit volumes of shape [geom.n_points]
                   key: 0 is nothing, 1 is signal, 2 is bkg
    :param geom: CTH geometry to draw
    """
    # Get the crystal indexes
    bkg_crystals = np.array([i for i, e in enumerate(output) if e == 2])
    sig_crystals = np.array([i for i, e in enumerate(output) if e == 1])
    trig_crystals = np.array([i for i, e in enumerate(trig_out) if e == 1])
    # Plot seperate by triggered and untriggered
    # Loop through and draw all crystals
    fill = False
    alpha = 1.0
    lw = 0.4
    for crystal in bkg_crystals:
        fill = False
        alpha = 1.0
        if np.in1d(crystal, trig_crystals):
            fill = True
            alpha = 0.5
        plot_add_cth_crystal(crystal, geom, alpha=alpha,
                             lw=lw, color="Red", fill=fill, **kwargs)
    for crystal in sig_crystals:
        fill = False
        alpha = 1.0
        if np.in1d(crystal, trig_crystals):
            fill = True
            alpha = 0.5
        plot_add_cth_crystal(crystal, geom, alpha=alpha,
                             lw=lw, color="Blue", fill=fill, **kwargs)

def plot_occupancies(sig_occu, back_occu, occu,
                     n_vols=4482,
                     x_pos=0.5, y_pos=0.5):
    vol_norm = n_vols / 100.
    ax1, _ = plot_feature(sig_occu, back_occu,
                            xlabel="Number of Wires Filled",
                            ylabel="Number of Events",
                            normed=False)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(np.round(ax1.get_xticks() / vol_norm, decimals=2))
    ax2.set_xlabel("Occupancy [%]")

    # Average signal
    a_sig = np.average(sig_occu)
    a_bkg = np.average(back_occu)
    a_occ = np.average(occu)
    textstr = 'Average Number of Wires'+\
              '\n  Signal:          {:.3g}'.format(a_sig) +\
              '\n  Background: {:.3g}'.format(a_bkg) +\
              '\n  Combined:    {:.3g}'.format(a_occ)

    # Average signal
    textstr += '\nAverage Occupancy'+\
               '\n  Signal:          {:.3g}%'.format(a_sig/vol_norm) +\
               '\n  Background: {:.3g}%'.format(a_bkg/vol_norm) +\
               '\n  Combined:    {:.3g}%'.format(a_occ/vol_norm)
    # these are matplotlib.patch.Patch properties
    _ = dict(facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax1.text(x_pos, y_pos, textstr, transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='left',
             fontsize=13)

def plot_feature_importance(classifier_and_feature,
                            feat_label_dict=None,
                            font_size=None):
    classifier, features = classifier_and_feature
    importances = classifier.feature_importances_
    feat_import = pandas.DataFrame({'effect': importances,
                                    'features': features})
    feat_import = feat_import.sort_values("effect", ascending=True)
    axis = plt.subplot(111)
    axis.barh(np.arange(len(feat_import.effect)), feat_import.effect, zorder=10)
    axis.set_yticks(np.arange(len(feat_import.values)))
    axis.minorticks_on()
    axis.set_xlabel("Normalized Relative Usage")
    axis.set_title("Feature Importance")
    axis.autoscale()
    axis.grid(b=True, which='major',
              axis='x', color='grey', linestyle='--', lw=2)
    axis.grid(b=True, which='minor',
              axis='x', color='grey', linestyle=':', lw=2)
    if feat_label_dict is None:
        feat_label_dict = dict()
    ordered_labels = OrderedDict((key, feat_label_dict.get(key, key))
                                 for key in feat_import.features)
    axis.set_yticklabels(ordered_labels.values(), fontsize=font_size)
    #axis.set_xticks([tick for tick in axis.get_xticks()[::2]])
    axis.set_xticklabels(["{:.0f}%".format(tick*100)
                          for tick in axis.get_xticks()], fontsize=font_size)
    return axis
