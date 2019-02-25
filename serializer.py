"""Author: Brandon Trabucco, Copyright 2019
Extracts the google contextual captions dataset into tf records.
MIT License"""

import tensorflow as tf
import csv
import collections
import string
import urllib
import base64
import time
import sys
import threading
import os

PUNCTUATION = string.punctuation
UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase
DIGITS = string.digits

TrainingExample = collections.namedtuple("TrainingExample", ("image_id", "data", "caption"))

def process_string(input_string):
    """Cleans up the input string for better processing.
    Args: input_string: a string that needs to be cleaned.
    Returns: the cleaned string in tokenozed form. """
    stage_one = ""
    for character in input_string:
        if character in PUNCTUATION:
            stage_one += " " + character + " "
        if character in UPPER:
            stage_one += character.lower()
        if character in LOWER + DIGITS + " ":
            stage_one += character
    stage_two = stage_one.replace("  ", " ").replace("  ", " ").strip()
    return stage_two.split(" ")

def process_url(image_url):
    """Load an image into bytes from a specified url.
    Args: image_url: the string url from which to fetch the image.
    Returns: the byte string corresponding to the encoded image."""
    return base64.b64encode(urllib.request.urlopen(image_url).read())

def process_tsv(tsv_filename, training_example_queue, max_queue_size, is_complete_flag):
    """Downloads images from the internet based on the TSV file specified.
    Args: tsv_filename: string, the location to the TSV file.
          training_example_queue: a queue of TrainingExamples to be processed.
          max_queue_size: an integer maximum capacity of the queue.
          is_complete_flag: an atopmic wrapper around a boolean flag signaling completion."""
    is_complete_flag[0] = False
    print("Starting to read the tsv file.")
    with open(tsv_filename, "r") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for i, row in enumerate(tsv_reader):
            caption, image_url = row
            training_example = TrainingExample(i, image_url, caption)
            while len(training_example_queue) >= max_queue_size:
                time.sleep(.5)
            training_example_queue.append(training_example)
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
            caption, image_data = process_string(training_example.caption), process_url(training_example.data)
        except urllib.error.HTTPError as e:
            print("Skipping bad training example, http address does not exist.")
            continue
        except urllib.error.URLError as e:
            print("Skipping bad training example, http address does not exist.")
            continue
        except Exception as e:
            print("Skipping bad training example.")
            continue
        training_example = TrainingExample(training_example.image_id, image_data, caption)
        while len(processed_queue) >= max_queue_size:
            time.sleep(.5)
        processed_queue.append(training_example)
        is_complete_flag[0] = len(training_example_queue) == 0 and no_more_inputs_flag[0]
        print("Finished processing a training example.")

def to_sequence_example(training_example):
    """Builds a SequenceExample proto for an image-caption pair.
    Args: training_example: A TrainingExample object.
    Returns: a string that represents the serialized sequence example."""
    return tf.train.SequenceExample(
        context=tf.train.Features(feature={
            "image/image_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[training_example.image_id])),
            "image/data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[training_example.data])),
        }), feature_lists=tf.train.FeatureLists(feature_list={
            "image/caption": tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[bytes(word, "UTF-8")])) for word in training_example.caption])
        })).SerializeToString()

def serialize_training_examples(name, output_dir, num_examples_per_file, training_example_queue, num_files_so_far, num_examples_so_far, no_more_inputs_flag, is_complete_flag):
    """Downloads images from the internet based on the TSV file specified.
    Args: name: a string name to prepend to each file.
          output_dir: a string path to the folder where files shall be saved.
          num_examples_per_file: an integer number of examples to save per file.
          training_example_queue: a queue of TrainingExamples to be processed.
          num_files_so_far: an atomic wrapper around an integer count for the number of files that have been processed.
          num_examples_so_far: an atomic wrapper around an integer count of the number of training examples processed so far.
          no_more_inputs_flag: an atopmic wrapper around a boolean flag signaling no more inputs to process.
          is_complete_flag: an atopmic wrapper around a boolean flag signaling completion."""
    is_complete_flag[0] = False
    while not is_complete_flag[0]:
        num_examples_in_this_file = 0
        print("Starting to write a file.")
        output_filename = "%s-%.7d.tfrecord" % (name, num_files_so_far[0])
        num_files_so_far[0] += 1
        writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, output_filename))
        while num_examples_in_this_file < num_examples_per_file and not is_complete_flag[0]:
            while len(training_example_queue) == 0 and not no_more_inputs_flag[0]:
                time.sleep(.5)
            training_example = training_example_queue.pop(0)
            serialized_example = to_sequence_example(training_example)
            writer.write(serialized_example)
            num_examples_so_far[0] += 1
            print("Wrote {0} examples so far.".format(num_examples_so_far[0]))  
            num_examples_in_this_file += 1
            is_complete_flag[0] = len(training_example_queue) == 0 and no_more_inputs_flag[0]
        writer.close()
        sys.stdout.flush()
        print("Finished writing a file.")

def process_dataset(tsv_filename, name, output_dir, num_examples_per_file, num_cores):
    """Serializes the dataset using the tsv entries
    Args: tsv_filename: string, the location to the TSV file.
          name: a string name to prepend to each file.
          output_dir: a string path to the folder where files shall be saved.
          num_examples_per_file: an integer number of examples to save per file.
          num_cores: an integer the number of physical cores on the computer."""
    queue_one, queue_two, tsv_finished, processor_finished, serializer_finished = [], [], [False], [False], [False]
    num_files_so_far, num_examples_so_far = [0], [0]
    tsv_thread = threading.Thread(target=process_tsv, args=(tsv_filename, queue_one, 2048, tsv_finished))
    processor_threads = [threading.Thread(target=process_training_examples, args=(queue_one, queue_two, 2048, 
        tsv_finished, processor_finished)) for i in range(num_cores)]
    serializer_threads = [threading.Thread(target=serialize_training_examples, args=(name, output_dir, 
        num_examples_per_file, queue_two, num_files_so_far, num_examples_so_far, processor_finished, serializer_finished)) for i in range(num_cores)]
    print("Starting all threads.")
    tsv_thread.start()
    for t in processor_threads + serializer_threads:
        t.start()
    coord = tf.train.Coordinator()
    coord.join([tsv_thread] + processor_threads + serializer_threads)
    print("All threads finished processing.")

if __name__ == "__main__":
    training_filename = "./Train_GCC-training.tsv"
    process_dataset(training_filename, "train", "./", 5096, 8)
    validation_filename = "./Validation_GCC-1.1.0-Validation.tsv"
    process_dataset(validation_filename, "val", "./", 1024, 8)
