
# coding: utf-8

# # Move Data to TFRecord

# In[1]:

import sys
sys.path.insert(0, '../modules')
from hits import CDCHits, CTHHits, CDCHits, FlatHits
from memory_profiler import memory_usage
from pprint import pprint
from collections import Counter
from root_numpy import list_branches
get_ipython().magic('load_ext memory_profiler')
from tracking import HoughSpace
from scipy import sparse
from tracking import HoughTransformer, HoughShifter
from cylinder import CDC
import scipy


# # Convenience Functions

# In[2]:

# The most common are stored in these notebooks
get_ipython().magic('run visualizations.ipynb')
get_ipython().magic('run data_tools.ipynb')


# In[3]:

def set_additional_branches(sample, row_name=None, cell_id=None, relative_time=None):
    """
    Set the trigger time and cell ID branches
    """
    if row_name:
        sample.data[row_name] = sample.geom.get_layers(sample.data[sample.flat_name])
    if cell_id:
        sample.data[cell_id] = sample.geom.get_indexes(sample.data[sample.flat_name])
    if relative_time:
        sample.data[relative_time] = sample.data[sample.time_name] - sample.data[sample.trig_name]


# ## Access data

# In[4]:

def get_measurment_and_neighbours(hit_sample, measurement, events=None):
    """
    Get the measurement on the wire and its neighbours in a classification-friendly way
    
    :return: a list of three numpy arrays of measurement 1) on wire, 2) to left, 3) to right
    """
    return [hit_sample.get_measurement(measurement, 
                                       events, 
                                       shift=i, 
                                       only_hits=True, 
                                       flatten=True) 
                for i in [0,-1,1]]


# ## Import the Signal Hits

# In[5]:

def test_labelling(hit_sample, sig_name, momentum_name, value):
    current_labels = hit_sample.get_events()[sig_name]
    momentum_magnitude = np.sqrt(np.square(hit_sample.get_events()[momentum_name+'.fX']) +                                 np.square(hit_sample.get_events()[momentum_name+'.fY']) +                                 np.square(hit_sample.get_events()[momentum_name+'.fZ']))
    pid_values = hit_sample.get_events()[pid_name]
    new_labels = np.logical_and(momentum_magnitude > value, pid_values == 11)
    print("Number of signal now : {}".format(sum(current_labels)))
    print("Number of signal actual : {}".format(sum(new_labels)))
    print("Number mislabelled : {}".format(hit_sample.n_hits - sum(current_labels == new_labels)))


# ### Make cuts

# In[6]:

def remove_coincidence(hit_samp, remove_hits=True):
    # Sort by local score name
    hit_samp.sort_hits(lcl_scr_name, ascending=False)
    all_hits_keep = hit_samp.get_measurement(hit_samp.hits_index_name, only_hits=True)
    # Make a mask   
    hit_samp.data[take_hit_name][all_hits_keep.astype(int)] = 1
    # Remove the hits
    if remove_hits:
        hit_samp.trim_hits(take_hit_name, values=1)
        hit_samp.sort_hits(hit_samp.flat_name)


# ## Define Our Samples

# In[7]:

# Define some branches to import
## Existing branches
prefix = "CDCHit.f"
drift_name = prefix + "DriftTime"
track_id_name = prefix + "Track.fTrackID"


## Branches to be filled
row_name = prefix +"Layers"
cell_id_name = prefix + "CellID"
rel_time_name = prefix + "Relative_Time"
take_hit_name = prefix + "Take_Hit"
lcl_scr_name = prefix + "Local_Score"
ngh_scr_name = prefix + "Neigh_Score"
hgh_scr_name = prefix + "Hough_Score"
trk_scr_name = prefix + "Track_Score"

empty_branches = [row_name, 
                  cell_id_name,
                  rel_time_name]

drift_dist_name = prefix + "DriftDist"
turn_id_name = prefix + "TurnID"
pid_name = prefix + "Track.fPID"
parent_track_id_name = prefix + "Track.fParentTrackID"
all_momentum_names = [ prefix + "Track.f" + st_sp + "Momentum.f" + coor 
                       for st_sp in ["Start", "Stop"] for coor in ["X", "Y", "Z"] ]
all_pos_names = [ prefix + "Track.f" + st_sp + "PosGlobal.f" + coor 
                       for st_sp in ["Start", "Stop"] for coor in ["P.fX", "P.fY", "P.fZ", "E"] ]
hit_pos_names = [ prefix + "MCPos.f" + coor for coor in ["P.fX", "P.fY", "P.fZ", "E"] ]
hit_mom_names = [ prefix + "MCMom.f" + coor for coor in ["X", "Y", "Z"] ]

# For track fitting
truth_branches = [turn_id_name] +                   hit_mom_names +                   hit_pos_names


# In[8]:

these_branches = dict()
these_branches["CDC"] = [drift_name, 
                         track_id_name] + truth_branches
these_branches["CTH"] = ["MCPos.fP.fX", "MCPos.fP.fY", "MCPos.fP.fZ"]


# In[9]:

file_root = "/home/five4three2/development/ICEDUST/"            "track-finding-yandex/data/"
file_root = "~/development/ICEDUST/track-finding-yandex/data/MC4p/"
sig_samples = ["oa_xx_xxx_09010000-0000_xerynzb6emaf_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09110000-0000_2mdcao2ehzya_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09210000-0000_opfmr3awxs2m_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09310000-0000_v62e3u5ppkju_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09410000-0000_z2p5ysva45vx_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09510000-0000_3eox62hw5ygi_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09610000-0000_7ctgq54tptae_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09710000-0000_kah3t5htgouf_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09810000-0000_a4tlhqvqnv4p_user-TrkTree_000_500signal-label.root",
               "oa_xx_xxx_09910000-0000_h6g347twij7d_user-TrkTree_000_500signal-label.root"]
sig_samples = [file_root + this_file for this_file in sig_samples]


# ## Import the Data

# In[24]:

get_ipython().magic('run data_tools.ipynb')


# In[11]:

train = data_import_file(sig_samples[-1],
                         use_cuts=["500","Trig","Track"],
                         branches=these_branches,
                         empty_branches=empty_branches)
train.set_trigger_time()
set_additional_branches(train.cdc, 
                        relative_time=rel_time_name,
                        row_name=row_name, 
                        cell_id=cell_id_name)

# Relabel position
pos = list()
pos += ["MCPos.fP.fX"]
pos += ["MCPos.fP.fY"]
pos += ["MCPos.fP.fZ"]

p = dict()
p["cdc"] = dict()
p["cdc"]["x"] =  train.cdc.prefix + pos[2]
p["cdc"]["y"] =  train.cdc.prefix + pos[1]
p["cdc"]["z"] =  train.cdc.prefix + pos[0]

p["cth"] = dict()
p["cth"]["x"] =  train.cth.prefix + pos[2]
p["cth"]["y"] =  train.cth.prefix + pos[1]
p["cth"]["z"] =  train.cth.prefix + pos[0]

train.cdc.data[p["cdc"]["x"]] = - (train.cdc.data[p["cdc"]["x"]]/10. - 765)
train.cdc.data[p["cdc"]["y"]] = train.cdc.data[p["cdc"]["y"]]/10.
train.cdc.data[p["cdc"]["z"]] = (train.cdc.data[p["cdc"]["z"]]/10. - 641)

train.cth.data[p["cth"]["x"]] = - (train.cth.data[p["cth"]["x"]]/10. - 765)
train.cth.data[p["cth"]["y"]] = train.cth.data[p["cth"]["y"]]/10.
train.cth.data[p["cth"]["z"]] = (train.cth.data[p["cth"]["z"]]/10. - 641)

# Relabel Momentum
mom = list()
mom += ["MCMom.fX"]
mom += ["MCMom.fY"]
mom += ["MCMom.fZ"]

m = dict()
m["cdc"] = dict()
m["cdc"]["x"] =  train.cdc.prefix + mom[2]
m["cdc"]["y"] =  train.cdc.prefix + mom[1]
m["cdc"]["z"] =  train.cdc.prefix + mom[0]

train.cdc.data[m["cdc"]["x"]] = -train.cdc.data[m["cdc"]["x"]]


# ## Remove Coincidence

# In[12]:

train.cdc.sort_hits(rel_time_name)


# In[ ]:

# Choose a sample event
hit_ids_mulit = None
for evt in range(10,11):
    # Initialize the fixed length array
    number_layers = 18
    max_number_wires = 300
    number_channels = 2
    an_array = np.zeros((18, 300, 2))
    
    # Get the wire and layer ids
    hit_ids = train.cdc.event_to_hits[evt]
    wire_ids = train.cdc.data[train.cdc.flat_name][hit_ids]
    hit_and_wire_ids = np.array([hit_ids, wire_ids]).T
    layer_ids = train.cdc.geom.get_layers(wire_ids)
    cell_ids = train.cdc.geom.get_indexes(wire_ids)
    
    # Get the hit times
    turn_id = train.cdc.get_measurement(turn_id_name, events=evt)
    wires_and_counts = np.array(np.unique(wire_ids, return_counts=True)).T
    all_data = None
    if False:
        pprint(train.cdc.data[train.cdc.edep_name][1170])
        pprint(train.cdc.data[rel_time_name][1170])
        pprint(train.cdc.data[train.cdc.flat_name][1170])
    if len(np.unique(wire_ids)) != len(wire_ids):
        print(len(np.unique(wire_ids)), len(wire_ids))
        multi_wires = wires_and_counts[[wires_and_counts[:,1] != 1]][:,0]
        multi_hits = np.array([hit_id for hit_id, wire_id in hit_and_wire_ids if wire_id in multi_wires])
        #train.cdc.data[1170][train.cdc.hit_type_name] = 0.
        #train.cdc.data[1068][train.cdc.hit_type_name] = 0.
        #train.cdc.data[1151][train.cdc.hit_type_name] = 0.
        #train.cdc.data[1096][train.cdc.hit_type_name] = 0.
        all_data = np.array([train.cdc.data[train.cdc.flat_name][multi_hits].astype(int),
                             multi_hits,
                             train.cdc.data[train.cdc.edep_name][multi_hits],
                             train.cdc.data[rel_time_name][multi_hits],
                             train.cdc.data[train.cdc.hit_type_name][multi_hits].astype(int)]).T
        all_data = all_data[all_data[:,0].argsort()]
        #pprint(all_data[:9,(0,2,3,4)])
        #hit_ids_multi = all_data[:9,1]
        #pprint(hit_ids_multi)


# In[50]:

get_ipython().magic('run data_tools.ipynb')


# In[51]:

data_remove_coincidence(train)


# In[56]:

hit_ids = train.cdc.event_to_hits[10]
new_all_data = np.array([train.cdc.data[train.cdc.flat_name][hit_ids].astype(int),
                         hit_ids,
                         train.cdc.data[train.cdc.edep_name][hit_ids],
                         train.cdc.data[rel_time_name][hit_ids],
                         train.cdc.data[train.cdc.hit_type_name][hit_ids].astype(int)]).T
new_all_data = new_all_data[new_all_data[:,0].argsort()]
pprint(new_all_data[:10,(0,2,3,4)])


# ## Move to fixed length arrays

# In[37]:

bkg_hit_evts = np.unique(train.cdc.get_background_hits()[samp.cdc.event_index_name])
no_bkg_hit_evts = np.setdiff1d(range(train.cdc.n_events), back_hit_events)
print(bkg_hit_evts.shape, no_bkg_hit_evts.shape, train.cdc.n_events)


# In[38]:

# Initialize the arrays
n_layers = 18
n_points = 300
n_channels = 2
flat_array = np.zeros((train.n_events, n_layers, n_points, n_channels))
# Fill the array
flat_array[train.cdc.get_events()[train.cdc.event_index_name].astype(int), 
           train.cdc.get_events()[row_name].astype(int),
           train.cdc.get_events()[cell_id_name].astype(int),
           :] = \
    np.array([train.cdc.get_events()[train.cdc.edep_name],
              train.cdc.get_events()[rel_time_name]]).T
shift = (300-train.cdc.geom.n_by_layer)
for i in range(18):
    flat_array[:,i,:,:] = np.roll(flat_array[:,i,:,:], shift[i])
flat_array = flat_array[no_bkg_hit_evts]


# In[42]:

figsize(60,4)
size=100
for evt in range(100):
    event = flat_array[evt, :, :, 1]
    positions = np.zeros((18, 300, 2))
    positions[:, :, 0] = np.vstack([range(300)] * 18)
    positions[:, :, 1] = np.vstack([range(18)] * 300).T
    hits = np.nonzero(event)
    plt.axis("off")
    plt.xlim(-1, 300)
    plt.ylim(-1, 18)
    plt.scatter(positions[:,:,0], positions[:,:,1], s=size, 
                marker='s', alpha=0.3, facecolors="none", edgecolor="black")
    plt.scatter(positions[hits][:,0], positions[hits][:,1], s=size, 
                marker='s', color="red", edgecolors="black")
    for i in range(17):
        plt.scatter(positions[i,-shift[i]/2:,0], positions[i,-shift[i]/2:,1], s=size, 
                    marker='s', alpha=0.3,color="black")
        plt.scatter(positions[i,:shift[i]/2,0], positions[i,:shift[i]/2,1], s=size, 
                    marker='s', alpha=0.3,color="black")
    image_name = "../images/2d_signal_"+str(evt)+".png"
    plt.savefig(image_name, bbox_inches="tight")
    plt.show()


# ## Visualize the data

# In[41]:

fig_s = (12.5,12.5)
s_scale = 35
plot_recbes = True
samp = train
figsize(60,4)
for evt in range(10):
    # Plot the output
    output = np.zeros(4482)
    geom_ids = samp.cdc.get_measurement(samp.cdc.flat_name, evt).astype(int)
    output[geom_ids] = 1
    cut = output
    plot_output(samp.cdc.get_hit_types(evt),
                samp.cdc.geom, size=output*s_scale, figsize=fig_s)
    # Add volume outlines
    plot_add_cth_outlines(samp.cth.geom)
    # Add the CTH vols with hits
    cth_vol_types = samp.cth.get_vol_types(evt)
    plot_add_cth(cth_vol_types, samp.cth.get_trig_vector(evt)[0], samp.cth.geom)
    cth_hits = samp.cth.get_events(evt)
    cdc_hits = samp.cdc.get_events(evt)
    #plt.scatter(cth_hits[p["cth"]["x"]], 
    #            cth_hits[p["cth"]["y"]], 
    #            s=1, transform=gca().transData._b)
    #plt.scatter(cdc_hits[p["cdc"]["x"]], 
    #            cdc_hits[p["cdc"]["y"]], 
    #            s=1, color="black",
    #            transform=gca().transData._b)
    plt.show()
    
    #for a,b in [("x","y"), ("z","y"), ("x","z")]:
        #plt.scatter(cdc_hits[p["cdc"][a]], 
        #            cdc_hits[p["cdc"][b]], 
        #            s=1, color="black")
        #if a == "x":
        #    plt.scatter(cth_hits[p["cth"][a]], 
        #                cth_hits[p["cth"][b]], 
        #                s=1, color="green")
        #else:
        #    plt.scatter(cth_hits[p["cth"][a]], 
        #                cth_hits[p["cth"][b]], 
        #                s=1, color="green")
        #plt.xlabel(a)
        #plt.ylabel(b)
        #plt.show()
    event = flat_array[evt, :, :, 1]
    positions = np.zeros((18, 300, 2))
    positions[:, :, 0] = np.vstack([range(300)] * 18)
    positions[:, :, 1] = np.vstack([range(18)] * 300).T
    hits = np.nonzero(event)
    plt.axis("off")
    plt.xlim(-1, 300)
    plt.ylim(-1, 18)
    plt.scatter(positions[:,:,0], positions[:,:,1], s=size, 
                marker='s', alpha=0.3, facecolors="none", edgecolor="black")
    plt.scatter(positions[hits][:,0], positions[hits][:,1], s=size, 
                marker='s', color="blue", edgecolors="black")
    for i in range(17):
        plt.scatter(positions[i,-shift[i]/2:,0], positions[i,-shift[i]/2:,1], s=size, 
                    marker='s', alpha=0.3,color="black")
        plt.scatter(positions[i,:shift[i]/2,0], positions[i,:shift[i]/2,1], s=size, 
                    marker='s', alpha=0.3,color="black")
    image_name = "../images/2d_signal_"+str(evt)+".png"
    plt.savefig(image_name, bbox_inches="tight")
    plt.show()
    print("=====================================================================")


# In[ ]:



