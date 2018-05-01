import sys
sys.path.insert(0, '../modules')
from hits import CDCHits, CTHHits, CDCHits, FlatHits
from pprint import pprint
from collections import Counter
from root_numpy import list_branches
from tracking import HoughSpace
from scipy import sparse
from tracking import HoughTransformer, HoughShifter
from cylinder import CDC
import scipy
import glob
import tensorflow as tf
import numpy as np
import sys
from data_tools import *


# In[2]:

def load_images(n_images, height=18, width=300, depth=2, n_filled=80):
    """
    Generate (n_images * height * width * channels) numpy array with 
    n_filled randomly filled entries per image.  Note that for a pixel, 
    both channels are either filled (randomly) or they are both empty.
    
    Parameters
    ----------
    n_images : int 
        Number of images
    height : int
        Height of each image
    width : int
        Width of each image
    depth : int
        Depth of each image
    n_filled : int
        Number of pixels filled in each event
    
    Returns
    -------
    images : ndarray
        Array of shape (n_images * height * width * channels)
    """
    # Initialize the return value
    image = np.zeros((n_images, height, width, 2))
    # Select around n_filled * n_images channels to fill
    layers = np.random.randint(0, high=height-1, size=(n_images, 80))
    cells = np.random.randint(0, high=width-1, size=(n_images, 80))
    # Fill the channels with random numbers
    image[:, layers, cells, :] = np.random.random(size=(n_images, 80,2))
    # Cast to 32 bits and return 
    return image.astype(np.float32)

def write_array_to_tfrecord(array, labels, filename, options=None):
    # Open TFRecords file, ensure we use gzip compression
    writer = tf.python_io.TFRecordWriter(filename, options=options)
    
    # Write all the images to a file
    for lbl, img in zip(labels, array):
        # Create a feature
        image_as_bytes = tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())])
        label_as_float = tf.train.FloatList(value=[lbl])
        feature = {'train/label':  tf.train.Feature(float_list=label_as_float),
                   'train/image':  tf.train.Feature(bytes_list=image_as_bytes)}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    # Close the writer and flush the buffer
    writer.close()
    sys.stdout.flush()
    
def read_tfrecord_to_array(filename, options=None):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.float32)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.float32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [18, 300, 2])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], 
                                            batch_size=1, 
                                            capacity=3,
                                            num_threads=1, 
                                            min_after_dequeue=2)
    return images, labels

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

# The most common are stored in these notebooks

# Define some branches to import
## Existing branches
prefix = "CDCHit.f"
drift_name = prefix + "DriftTime"
## Branches to be filled
row_name = prefix +"Layers"
cell_id_name = prefix + "CellID"
rel_time_name = prefix + "Relative_Time"
empty_branches = [row_name, 
                  cell_id_name,
                  rel_time_name]
# Branches we need
hit_pos_names = [prefix + "MCPos.f" + coor
                 for coor in ["P.fX", "P.fY", "P.fZ", "E"] ]
hit_mom_names = [prefix + "MCMom.f" + coor for coor in ["X", "Y", "Z"] ]
# For track fitting
truth_branches = hit_mom_names + hit_pos_names
these_branches = dict()
these_branches["CDC"] = [drift_name, track_id_name] + truth_branches

# Open all of our files
file_root = "/home/five4three2/development/ICEDUST/track-finding-yandex/data/"
file_root = "~/development/ICEDUST/track-finding-yandex/data/MC4p/"
sig_samples = glob.glob(file_root+"/oa_xx_xxx_*500*root")
print(sig_samples)
assert 0

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

## Remove Coincidence
train.cdc.sort_hits(rel_time_name)
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

# TODO OPEN FOR LOOP OVER FILES HERE
if False:
    # Set the number of samples
    n_random_samples = 10
    original_images = load_images(n_random_samples)
    original_labels = np.random.random(n_random_samples)
    compression = tf.python_io.TFRecordCompressionType.GZIP
    tf_io_opts = tf.python_io.TFRecordOptions(compression)
    # Write the file
    write_array_to_tfrecord(original_images, original_labels, "train.tfrecords", tf_io_opts)
    # Read the files
    new_images, new_labels = [], []
    with tf.Session() as sess:
        # Get the images and labels
        tf_images, tf_labels = read_tfrecord_to_array("train.tfrecords", tf_io_opts)
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for batch_index in range(n_random_samples):
            img, lbl = sess.run([tf_images, tf_labels])
            new_images += [img]
            new_labels += [lbl]
    
        # Stop the threads
        coord.request_stop()
    
        # Wait for threads to stop
        coord.join(threads)
        sess.close()
    # Compare the two arrays
    np.testing.assert_allclose(original_images, np.vstack(new_images), rtol=1e-7)



