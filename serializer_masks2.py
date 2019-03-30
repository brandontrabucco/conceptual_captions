"""Author: Brandon Trabucco, Copyright 2019
Serializes the conceptual captions dataset into tf records."""


import tensorflow as tf
import math
import numpy as np
import pickle as pkl
import os
import time
import collections
import sys
import threading
import multiprocessing
import FasterRCNN.inference
from FasterRCNN.inference import create_r101fpn_mask_rcnn_model_graph as create_model
from nltk.tokenize import wordpunct_tokenize as word_tokenize


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("image_folder", "./train_images", "Path to the image files.")
tf.flags.DEFINE_string("tsv_filename", "./Train_GCC-training.tsv", "Path to the TSV file.")
tf.flags.DEFINE_string("vocab_filename", "./word.vocab", "Path to the vocab file.")
tf.flags.DEFINE_string("output_dir", "./train_mask_tfrecords/", "Output data directory.")
tf.flags.DEFINE_integer("num_threads", 16, "Number of threads to use when serializing the dataset.")
tf.flags.DEFINE_integer("num_examples_per_file", 5096, "Number of threads to use when serializing the dataset.")
tf.flags.DEFINE_integer("queue_size", 64, "The number of examples to extract at once.")
tf.flags.DEFINE_integer("image_size", 600, "The size of the image smaller dimension.")
tf.flags.DEFINE_integer("start_at_file_index", 0, "The number of tfrecords to skip.")
tf.flags.DEFINE_string("visible_gpus", "0,0,1", "Which GPU to use.")
tf.flags.DEFINE_string("gpus_memory", "0.4,0.4,1", "Which GPU to use.")
FLAGS = tf.flags.FLAGS


TrainingExample = collections.namedtuple("TrainingExample", (
    "image_id", "image", "caption", "boxes", "masks", "labels",
    "r1", "r2", "r3", "r4", "m1", "m2", "m3", "m4"))


def rescale_shorter_edge(images, new_shorter_edge):
    def compute_longer_edge(height, width, new_shorter_edge):
        return tf.cast(width * new_shorter_edge / height, tf.int32)
    height, width = tf.shape(images)[0], tf.shape(images)[1]
    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, compute_longer_edge(height, width, new_shorter_edge)),
        lambda: (compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge))
    return tf.image.resize_images(images, new_height_and_width)


class Extractor(object):
    """Helper class for extracting features from images."""
    def __init__(self, list_of_images, gpu_id=0, memory_fraction=1):
        """Creates handles to the TensorFlow computational graph."""
        self.list_of_images = list_of_images
        self.list_of_filenames = []
        for x in list_of_images:
            self.list_of_filenames.append(x.image)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_tensor = self.get_image_tensor(self.list_of_filenames, gpu_id)
            # Load the image tensor using a queue for massive parallelization.
            batch_size = tf.shape(self.image_tensor)[0]
            # Create a single TensorFlow Session for all image decoding calls.
            self.sess = tf.Session(
                graph=self.graph,
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(
                        visible_device_list=str(gpu_id),
                        per_process_gpu_memory_fraction=memory_fraction,
                        )))
            # Creates the mask rcnn model
            self.fetch_dict = create_model(self.image_tensor, self.sess)
            self.boxes = self.fetch_dict["boxes"]
            self.labels = self.fetch_dict["labels"]
            self.masks = self.fetch_dict["masks"]
            self.r1 = self.fetch_dict["region_features_1"]
            self.r2 = self.fetch_dict["region_features_2"]
            self.r3 = self.fetch_dict["region_features_3"]
            self.r4 = self.fetch_dict["region_features_4"]
            self.m1 = self.fetch_dict["mask_features_1"]
            self.m2 = self.fetch_dict["mask_features_2"]
            self.m3 = self.fetch_dict["mask_features_3"]
            self.m4 = self.fetch_dict["mask_features_4"]
            # store a handle to how many images were extracted so far
            self.num_images_so_far = tf.Variable(0)
            self.increment_by_batch_size = tf.assign(self.num_images_so_far, self.num_images_so_far + batch_size)
            self.sess.run(tf.variables_initializer([self.num_images_so_far]))
    def get_image_tensor(self, list_of_filenames, gpu_id):
        """Loads the images using the CPU and coppies them to the GPU.
        Args:
            list_of_bytes: a list of filename strings for encoded images.
        Returns:
            a batch representing an image tensor."""
        # Create a queue of strings
        dataset = tf.data.Dataset.from_tensor_slices(list_of_filenames)
        # Create a lazy image loading function to save memory.
        def lazy_load_image(image_filename):
            with tf.gfile.FastGFile(image_filename, "rb") as f:
                return f.read()
        # Convert from a filename to an image tensor.
        def filename_to_tensor(image_filename):
            image_bytes = tf.py_func(lazy_load_image, [image_filename], tf.string)
            image_tensor = tf.image.decode_jpeg(image_bytes, channels=3)
            return rescale_shorter_edge(image_tensor, FLAGS.image_size)
        # Convert each element to a tensor
        dataset = dataset.map(filename_to_tensor, num_parallel_calls=16)
        dataset = dataset.apply(tf.contrib.data.ignore_errors())
        # Create a batch with the specified size
        dataset = dataset.batch(1)
        def prepare_final_batch(image_tensor):
            return tf.transpose(tf.cast(image_tensor, tf.float32), [0, 3, 1, 2])
        # Copy the data to the GPU ahead of time
        dataset = dataset.map(prepare_final_batch, num_parallel_calls=16)
        dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=2))
        image_tensor = dataset.make_one_shot_iterator().get_next()
        return image_tensor
    def get_num_extracted_so_far(self):
        """Extracts a single batch of image features.
        Returns: 
            a integer representing the number of examples processed."""
        return int(self.sess.run(self.num_images_so_far))
    def extract(self):
        """Extracts a single batch of image features.
        Returns: 
            a list of ImageMetadata objects."""
        position_before = self.get_num_extracted_so_far()
        (boxes, masks, labels, 
            r1, r2, r3, r4, m1, m2, m3, m4, _) = self.sess.run([
            self.boxes, self.masks, self.labels, 
            self.r1, self.r2, self.r3, self.r4,
            self.m1, self.m2, self.m3, self.m4,
            self.increment_by_batch_size])
        position_after = self.get_num_extracted_so_far()
        list_of_preextracted_images = []
        for i in range(position_after - position_before):
            image_meta = self.list_of_images[i + position_before]
            list_of_preextracted_images.append(
                TrainingExample(
                    image_id=image_meta.image_id,
                    image=image_meta.image,
                    caption=image_meta.caption,
                    boxes=boxes,
                    masks=masks,
                    labels=labels,
                    r1=r1, r2=r2, r3=r3, r4=r4, 
                    m1=m1, m2=m2, m3=m3, m4=m4))
        return list_of_preextracted_images


def process_string(input_string, vocab):
    stage_two = word_tokenize(input_string.strip().lower())
    return [vocab[word] if word in vocab else vocab["<unk>"] for word in (["<s>"] + stage_two + ["</s>"])]


def get_all_training_examples(image_folder, tsv_filename):
    start_time = time.time()
    training_examples = []
    with open(tsv_filename, "r", encoding="utf-8") as tsv_file:  
        for i, line in enumerate(tsv_file):
            training_examples.append(
                TrainingExample(
                    i, os.path.join(image_folder, "{0}.jpg".format(i)), 
                    line.strip().split("\t")[0], None, None, None,
                        None, None, None, None, None, None, None, None))
    end_time = time.time()
    print("Finished loading training examples, took {0} seconds.".format(end_time - start_time))
    return training_examples


def to_sequence_example(metadata, image_bytes, vocab):
    num_regions = metadata.boxes.shape[0]
    return tf.train.SequenceExample(
        context=tf.train.Features(feature={
            "image/image_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[metadata.image_id])),
            "image/data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        }), feature_lists=tf.train.FeatureLists(feature_list={
            "image/caption": tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(
                value=[word])) for word in process_string(metadata.caption, vocab)]),
            "image/labels": tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(
                value=metadata.labels[i:(i + 1)].ravel())) for i in range(num_regions)]),
            "image/boxes": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.boxes[i, :].ravel())) for i in range(num_regions)]),
            "image/masks": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.masks[i, :, :].ravel())) for i in range(num_regions)]),
            "image/r1": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.r1[i, :].ravel())) for i in range(num_regions)]),
            "image/r2": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.r2[i, :].ravel())) for i in range(num_regions)]),
            "image/r3": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.r3[i, :].ravel())) for i in range(num_regions)]),
            "image/r4": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.r4[i, :].ravel())) for i in range(num_regions)]),
            "image/m1": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.m1[i, :].ravel())) for i in range(num_regions)]),
            "image/m2": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.m2[i, :].ravel())) for i in range(num_regions)]),
            "image/m3": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.m3[i, :].ravel())) for i in range(num_regions)]),
            "image/m4": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(
                value=metadata.m4[i, :].ravel())) for i in range(num_regions)]),
        })).SerializeToString()


def write_all_training_examples(training_examples, vocab_filename, output_dir, num_files_so_far, 
        num_examples_per_file, queue_is_finished):
    start_time = time.time()
    print("Starting to write training examples.")
    with open(vocab_filename, "rb") as f:
        vocab = {word: i for i, word in enumerate(list(pkl.load(f)) + ["<s>", "</s>", "<unk>"])}
    while len(training_examples) > 0 or not all(queue_is_finished):
        num_examples_in_this_file = 0
        print("Starting to write a file.")
        output_filename = "%.5d.tfrecord" % (num_files_so_far[0])
        num_files_so_far[0] += 1
        writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, output_filename))
        while num_examples_in_this_file < num_examples_per_file and (len(training_examples) > 0 or not all(queue_is_finished)):
            try:
                x = training_examples.pop(0)
            except Exception as e:
                time.sleep(.5)
                continue
            with tf.gfile.FastGFile(x.image, "rb") as f:
                image_bytes = f.read()
            writer.write(to_sequence_example(x, image_bytes, vocab))
            num_examples_in_this_file += 1
        writer.close()
        sys.stdout.flush()
        print("Finished writing a file.")
    end_time = time.time()
    print("Finished writing training examples, took {0} hours.".format((end_time - start_time) / 3600))


def launch_extractor_thread(thread_id, input_metadata, output_metadata, total_count, queue_is_finished, gpu_id, memory_fraction):
    print("Loading images for the Extractor.")
    try:
        extractor = Extractor(input_metadata, gpu_id=gpu_id, memory_fraction=memory_fraction)
    except Exception as e:
        print("An error was encountered when creating extractor: {0}".format(str(e)))
        exit()
    print("Starting the GPU thread for Extractor.")
    time_since_last_extract = time.time()
    num_extracted_so_far = 0
    while time.time() - time_since_last_extract < 3600.0 and num_extracted_so_far < total_count:
        if len(output_metadata) >= FLAGS.queue_size:
            time.sleep(.5)
            continue
        try:
            buffer = extractor.extract()
            output_metadata.extend(buffer)
            num_extracted_so_far += len(buffer)
            print("Extractor {0} extracted {1:07d} examples: {2:03.3f}% complete: eta {3:05.3f} hours.".format(
                thread_id,
                num_extracted_so_far, num_extracted_so_far / total_count * 100.0, 
                (time.time() - time_since_last_extract) / len(buffer) * (total_count - num_extracted_so_far) / 3600.0 ))
            time_since_last_extract = time.time()
        except Exception as e:
            #print("An error was encountered when extracting extractor: {0}".format(str(e)))
            time.sleep(.1)
        queue_is_finished[thread_id] = len(input_metadata) == 0


if __name__ == "__main__":
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    training_examples = get_all_training_examples(FLAGS.image_folder, FLAGS.tsv_filename)
    training_examples = training_examples[(FLAGS.start_at_file_index * FLAGS.num_examples_per_file):]
    manager = multiprocessing.Manager()
    extracted_training_examples = manager.list()
    num_files_so_far = [FLAGS.start_at_file_index]
    gpu_ids = [int(x) for x in FLAGS.visible_gpus.split(",")]
    memory_fractions = [float(x) for x in FLAGS.gpus_memory.split(",")]
    queue_is_finished = [False for i in gpu_ids]
    split_length = math.ceil(len(training_examples) / len(gpu_ids))
    split_training_examples = [training_examples[i:i + split_length] for i in range(
        0, len(training_examples), split_length)]
    extractor_threads = [
        multiprocessing.Process(
            target=launch_extractor_thread, 
            args=(i, z, extracted_training_examples, 
                len(z), queue_is_finished, x, y)) for i, x, y, z in zip(
                    list(range(len(gpu_ids))), gpu_ids, memory_fractions, split_training_examples)]
    serializer_threads = [
        threading.Thread(
            target=write_all_training_examples, 
            args=(extracted_training_examples, FLAGS.vocab_filename, FLAGS.output_dir, num_files_so_far, 
                FLAGS.num_examples_per_file, queue_is_finished)) for i in range(FLAGS.num_threads)]
    for thread in extractor_threads:
        thread.start()
    for thread in serializer_threads:
        thread.start()
    coord = tf.train.Coordinator()
    coord.join(extractor_threads + serializer_threads)