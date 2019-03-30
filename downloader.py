"""Author: Brandon Trabucco, Copyright 2019
Extracts the google conceptual captions dataset into jpg files.
MIT License"""


import time
import os
import tensorflow as tf
import collections
import urllib
import threading


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("tsv_filename", "./Train_GCC-training.tsv", "Path to the TSV file.")
tf.flags.DEFINE_string("output_dir", "./images/", "Output data directory.")
tf.flags.DEFINE_integer("num_threads", 16, "Number of threads to use when downloading the dataset.")
FLAGS = tf.flags.FLAGS


TrainingExample = collections.namedtuple("TrainingExample", ("image_id", "image_url"))


def get_all_image_urls(tsv_filename):
    start_time = time.time()
    print("Starting to process image urls.")
    training_examples = []
    with open(tsv_filename, "r", encoding="utf-8") as tsv_file:  
        for i, line in enumerate(tsv_file):
            training_examples.append(TrainingExample(i, line.strip().split("\t")[1]))
    end_time = time.time()
    print("Finished processing image urls, took {0} seconds.".format(end_time - start_time))
    return training_examples


def download_and_save_all_images(training_examples, output_dir):
    start_time = time.time()
    print("Starting to download images from urls.")
    while len(training_examples) > 0:
        x = training_examples.pop(0)
        try:
            with open(os.path.join(output_dir, "{0}.jpg".format(x.image_id)), "wb") as f:
                f.write(urllib.request.urlopen(x.image_url, timeout=10).read())
        except Exception as e:
            print("Skipping bad url {0} for image {1}.".format(x.image_url, x.image_id))
    end_time = time.time()
    print("Finished downloading images from urls, took {0} hours.".format((end_time - start_time) / 3600))


def watch_and_report_time(training_examples, total_length):
    start_time = time.time()
    while len(training_examples) > 0:
        time.sleep(10.0)
        print("Downloaded {0} images, {1} hours until complete.".format(
            total_length - len(training_examples),
            (len(training_examples)) * (time.time() - start_time) / (total_length - len(training_examples)) / 3600))


if __name__ == "__main__":
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    training_examples = get_all_image_urls(FLAGS.tsv_filename)
    downloader_threads = [
        threading.Thread(
            target=download_and_save_all_images, 
            args=(training_examples, FLAGS.output_dir)) for i in range(FLAGS.num_threads)]
    time_thread = threading.Thread(
            target=watch_and_report_time, 
            args=(training_examples, len(training_examples)))
    time_thread.start()
    for thread in downloader_threads:
        thread.start()
    coord = tf.train.Coordinator()
    coord.join([time_thread] + downloader_threads)

