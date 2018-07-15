import sys
from pprint import pprint
from collections import Counter
from scipy import sparse
import scipy
import glob
import tensorflow as tf
import numpy as np

LABEL_NAMES = ["p_t", "p_z", 
               "entry_x", "entry_y", "entry_z", 
               "vert_x", "vert_y", "vert_z",
               "n_turns"]

def load_images_and_labels(npz_file):
    # Initialize the return value
    loaded = np.load(npz_file)
    return loaded["image"].astype(np.float32), \
           loaded["labels"].astype(np.float32)

def write_array_to_tfrecord(array, labels, filename, options=None):
    # Open TFRecords file, ensure we use gzip compression
    writer = tf.python_io.TFRecordWriter(filename, options=options)
    
    # Write all the images to a file
    for lbl, img in zip(labels, array):
        # Create the feature dictionary and enter the image
        image_as_bytes = tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())])
        feature = {'image':  tf.train.Feature(bytes_list=image_as_bytes)}
        # Create anentry for each label
        for a_lab, name_lab in zip(lbl, LABEL_NAMES):
            label_as_float = tf.train.FloatList(value=[a_lab])
            feature[name_lab] = tf.train.Feature(float_list=label_as_float)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    # Close the writer and flush the buffer
    writer.close()
    sys.stdout.flush()
    
def read_tfrecord_to_array(filename, options=None):
    feature = {'image': tf.FixedLenFeature([], tf.string)}
    for name in LABEL_NAMES:
        feature[name] = tf.FixedLenFeature([], tf.float32)
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image'], tf.float32)
    # Cast label data into int32
    all_labels = tf.stack([tf.cast(features[name], tf.float32)
                           for name in LABEL_NAMES])
    # Reshape image data into the original shape
    image = tf.reshape(image, [18, 300, 2])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images, all_labels = tf.train.batch([image, all_labels], 
                                                 batch_size=1, 
                                                 capacity=3,
                                                 num_threads=1)
                                                 #min_after_dequeue=2)
    return images, all_labels

# Set the number of samples
filename = sys.argv[1]
original_images, original_labels = load_images_and_labels(filename)
compression = tf.python_io.TFRecordCompressionType.GZIP
tf_io_opts = tf.python_io.TFRecordOptions(compression)
# Write the file
out_file = filename.split("/")[-1].split(".")[0]+".tfrecord"
write_array_to_tfrecord(original_images, original_labels, out_file, tf_io_opts)
test_me = False
if test_me:
    # Read the files
    new_images, new_labels = [], []
    n_samps = 10
    with tf.Session() as sess:
        # Get the images and labels
        tf_images, tf_labels = read_tfrecord_to_array(out_file, tf_io_opts)
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for batch_index in range(n_samps):
            img, lbl = sess.run([tf_images, tf_labels])
            new_images += [img]
            new_labels += [lbl]
    
        # Stop the threads
        coord.request_stop()
    
        # Wait for threads to stop
        coord.join(threads)
        sess.close()
    
    # Compare the two arrays
    np.testing.assert_allclose(original_images[:n_samps], np.vstack(new_images),
                               rtol=1e-7, verbose=True)
    np.testing.assert_allclose(original_labels[:n_samps], np.vstack(new_labels),
                               rtol=1e-7, verbose=True)
