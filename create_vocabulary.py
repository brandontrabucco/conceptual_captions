"""Author: Brandon Trabucco, Copyright 2019
Creates a vocabulary from the conceptual captions dataset."""


import tensorflow as tf
import string
import pickle as pkl
import time
from nltk.tokenize import wordpunct_tokenize as word_tokenize


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("tsv_filename", "./Train_GCC-training.tsv", "Path to the TSV file.")
tf.flags.DEFINE_string("vocab_filename", "./word.vocab", "Path to the vocab file.")
tf.flags.DEFINE_integer("min_instances", 5, "Minimum instances of word before truncation.")
FLAGS = tf.flags.FLAGS


def process_string(input_string):
    return word_tokenize(input_string.strip().lower())


def get_unique_words(sentence_generator):
    start_time = time.time()
    unique_words = {}
    for line in sentence_generator:
        tokenized_sentence = process_string(line.strip().split("\t")[0])
        for word in tokenized_sentence:
            if word not in unique_words:
                unique_words[word] = 0
            unique_words[word] += 1
    end_time = time.time()
    print("Finished loading vocabulary, took {0} seconds.".format(end_time - start_time))
    return unique_words


def get_unique_words(tsv_filename):
    start_time = time.time()
    unique_words = {}
    with open(tsv_filename, "r", encoding="utf-8") as tsv_file:  
        for i, line in enumerate(tsv_file):
            tokenized_sentence = process_string(line.strip().split("\t")[0])
            for word in tokenized_sentence:
                if word not in unique_words:
                    unique_words[word] = 0
                unique_words[word] += 1
    end_time = time.time()
    print("Finished loading vocabulary, took {0} seconds.".format(end_time - start_time))
    return unique_words


if __name__ == "__main__":


    word_dict = get_unique_words(FLAGS.tsv_filename)
    word_list = list(zip(*list(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))))[0]
    for i, word in enumerate(word_list):
        if word_dict[word] < FLAGS.min_instances:
            word_list = word_list[:(i + 1)]
            break
    print("Created a vocabulary with {0} words.".format(len(word_list)))
    with open(FLAGS.vocab_filename, "wb") as f:
        pkl.dump(word_list, f)

