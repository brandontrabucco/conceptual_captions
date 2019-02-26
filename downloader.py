"""Author: Brandon Trabucco, Copyright 2019
Extracts the google contextual captions dataset into jpg files.
MIT License"""

import tensorflow as tf
import csv
import collections
import urllib
import time
import sys
import threading
import os

TrainingExample = collections.namedtuple("TrainingExample", ("image_id", "data", "caption"))

def process_url(image_url):
    """Load an image into bytes from a specified url.
    Args: image_url: the string url from which to fetch the image.
    Returns: the byte string corresponding to the encoded image."""
    return urllib.request.urlopen(image_url).read()

def process_tsv(tsv_filename, starting_index, training_example_queue, max_queue_size, is_complete_flag):
    """Downloads images from the internet based on the TSV file specified.
    Args: tsv_filename: string, the location to the TSV file.
          starting_index: an integer the number of images that were processed so far from tsv file.
          training_example_queue: a queue of TrainingExamples to be processed.
          max_queue_size: an integer maximum capacity of the queue.
          is_complete_flag: an atopmic wrapper around a boolean flag signaling completion."""
    is_complete_flag[0] = False
    print("Starting to read the tsv file.")
    with open(tsv_filename, "r") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        tsv_iterable = enumerate(tsv_reader).__iter__()
        while True:
            try:
                i, row = tsv_iterable.__next__()
                if i < starting_index:
                    continue
                caption, image_url = row
                training_example = TrainingExample(i, image_url, caption)
                while len(training_example_queue) >= max_queue_size:
                    time.sleep(.5)
                training_example_queue.append(training_example)
            except StopIteration as e:
                break
            except Exception as e:
                print("Skipping bad tsv row.")
                continue
    is_complete_flag[0] = True
    print("Finished reading the tsv file.")

def process_training_examples(training_example_queue, processed_queue, max_queue_size, no_more_inputs_flag, is_complete_flag):
    """Downloads images from the internet based on the TSV file specified.
    Args: training_example_queue: a queue of TrainingExamples to be processed.
          processed_queue: a queue of processed training examples.
          max_queue_size: an integer maximum capacity of the queue.
          no_more_inputs_flag: an atopmic wrapper around a boolean flag signaling no more inputs to process.
          is_complete_flag: an atopmic wrapper around a boolean flag signaling completion."""
    is_complete_flag[0] = False
    while not is_complete_flag[0]:
        print("Starting to process a training example.")
        while len(training_example_queue) == 0 and not no_more_inputs_flag[0]:
            time.sleep(.5)
        training_example = training_example_queue.pop(0)
        try:
            image_data = process_url(training_example.data)
        except Exception as e:
            print("Skipping bad training example.")
            continue
        training_example = TrainingExample(training_example.image_id, image_data, None)
        while len(processed_queue) >= max_queue_size:
            time.sleep(.5)
        processed_queue.append(training_example)
        is_complete_flag[0] = len(training_example_queue) == 0 and no_more_inputs_flag[0]
        print("Finished processing a training example.")

def save_training_examples(name, output_dir, training_example_queue, num_examples_so_far, no_more_inputs_flag, is_complete_flag):
    """Downloads images from the internet based on the TSV file specified.
    Args: name: a string name to prepend to each file.
          output_dir: a string path to the folder where files shall be saved.
          training_example_queue: a queue of TrainingExamples to be processed.
          num_examples_so_far: an atomic wrapper around an integer count of the number of training examples processed so far.
          no_more_inputs_flag: an atomic wrapper around a boolean flag signaling no more inputs to process.
          is_complete_flag: an atomic wrapper around a boolean flag signaling completion."""
    is_complete_flag[0] = False
    while not is_complete_flag[0]:
        while len(training_example_queue) == 0 and not no_more_inputs_flag[0]:
            time.sleep(.5)
        training_example = training_example_queue.pop(0)
        with open(os.path.join(output_dir, name + "{0}.jpg".format(training_example.image_id)), "wb") as f:
            f.write(training_example.data)
        num_examples_so_far[0] += 1
        print("Wrote {0} examples so far.".format(num_examples_so_far[0]))  
        is_complete_flag[0] = len(training_example_queue) == 0 and no_more_inputs_flag[0]

def process_dataset(tsv_filename, starting_index, name, output_dir, num_cores):
    """Serializes the dataset using the tsv entries
    Args: tsv_filename: string, the location to the TSV file.
          starting_index: an integer the number of images that were processed so far from tsv file.
          name: a string name to prepend to each file.
          output_dir: a string path to the folder where files shall be saved.
          num_examples_per_file: an integer number of examples to save per file.
          num_cores: an integer the number of physical cores on the computer."""
    queue_one, queue_two, tsv_finished, processor_finished, serializer_finished = [], [], [False], [False], [False]
    num_examples_so_far = [starting_index]
    tsv_thread = threading.Thread(target=process_tsv, args=(tsv_filename, starting_index, queue_one, 2048, tsv_finished))
    processor_threads = [threading.Thread(target=process_training_examples, args=(queue_one, queue_two, 2048, 
        tsv_finished, processor_finished)) for i in range(num_cores)]
    saver_threads = [threading.Thread(target=save_training_examples, args=(name, output_dir, 
        queue_two, num_examples_so_far, processor_finished, serializer_finished)) for i in range(num_cores)]
    print("Starting all threads.")
    tsv_thread.start()
    for t in processor_threads + saver_threads:
        t.start()
    coord = tf.train.Coordinator()
    coord.join([tsv_thread] + processor_threads + saver_threads)
    print("All threads finished processing.")

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("training_tsv", "./Train_GCC-training.tsv", "Path to the Training TSV file.")
tf.flags.DEFINE_string("validation_tsv", "./Validation_GCC-1.1.0-Validation.tsv", "Path to the Validation TSV file.")
tf.flags.DEFINE_string("output_dir", "./images/", "Output data directory.")
tf.flags.DEFINE_integer("num_cores", 8, "Number of cores to use when extracting the dataset.")
tf.flags.DEFINE_integer("training_starting_index", 0, "The number of images that have been downloaded so far.")
tf.flags.DEFINE_integer("validation_starting_index", 0, "The number of images that have been downloaded so far.")
FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    process_dataset(FLAGS.training_tsv, FLAGS.training_starting_index, "train", FLAGS.output_dir, FLAGS.num_cores)
    process_dataset(FLAGS.validation_tsv, FLAGS.validation_starting_index, "val", FLAGS.output_dir, FLAGS.num_cores)
