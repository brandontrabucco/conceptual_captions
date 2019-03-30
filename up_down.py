"""Author: Brandon Trabucco, Copyright 2019
Loads images from the conceptual captions dataset and trains a show and tell model."""


import tensorflow as tf
import pickle as pkl
import os
import time
import numpy as np
import itertools
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_101
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_arg_scope
import captionkit
from captionkit.up_down_cell import UpDownCell
from captionkit.image_captioner import ImageCaptioner
from captionkit.resnet_v2_101 import ResNet


def get_dataset(tfrecord_file_pattern, image_height, image_width, 
        num_epochs, batch_size, device="/gpu:0"):
    dataset = tf.data.Dataset.list_files(tfrecord_file_pattern).apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=batch_size, sloppy=True))
    def process_tf_record(x):
        context, sequence = tf.parse_single_sequence_example(x,
            context_features = {"image/data": tf.FixedLenFeature([], dtype=tf.string),
                "image/boxes": tf.VarLenFeature(tf.float32),
                "image/boxes/shape": tf.FixedLenFeature([2], dtype=tf.int64),
                "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
            sequence_features = {"image/caption": tf.FixedLenSequenceFeature([], dtype=tf.int64)})
        image, image_id, caption = context["image/data"], context["image/image_id"], sequence["image/caption"]
        boxes = tf.reshape(tf.sparse.to_dense(context["image/boxes"]), context["image/boxes/shape"]) / 512.0
        image = tf.image.resize_images(tf.image.convert_image_dtype(tf.image.decode_jpeg(
            image, channels=3), dtype=tf.float32), size=[image_height, image_width])
        input_length = tf.expand_dims(tf.subtract(tf.shape(caption)[0], 1), 0)
        return {"image": image, "image_id": image_id, 
            "boxes": boxes,
            "input_seq": tf.slice(caption, [0], input_length), 
            "target_seq": tf.slice(caption, [1], input_length), 
            "indicator": tf.ones(input_length, dtype=tf.int32)}
    dataset = dataset.map(process_tf_record, num_parallel_calls=batch_size)
    dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(batch_size * 10, count=num_epochs))
    padded_shapes = {"image": [image_height, image_width, 3], "image_id": [], 
        "boxes": [None, 4],
        "input_seq": [None], "target_seq": [None], "indicator": [None]}
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device, buffer_size=1))
    return dataset.make_initializable_iterator()


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_string("tfrecord_file_pattern", "../train_mask_tfrecords/?????.tfrecord", "Pattern of the TFRecord files.")
tf.flags.DEFINE_string("vocab_filename", "../word.vocab", "Path to the vocab file.")
tf.flags.DEFINE_integer("image_height", 256, "")
tf.flags.DEFINE_integer("image_width", 256, "")
tf.flags.DEFINE_integer("num_epochs", 10, "")
tf.flags.DEFINE_integer("batch_size", 100, "")
FLAGS = tf.flags.FLAGS


class Vocabulary(object):
    def __init__(self, vocab_names, start_word, end_word, unk_word):
        vocab = dict([(x, y) for (y, x) in enumerate(vocab_names)])
        print("Created vocabulary with %d names." % len(vocab_names))
        self.vocab = vocab
        self.reverse_vocab = vocab_names
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]
    def word_to_id(self, word):
        if isinstance(word, list):
            return [self.word_to_id(w) for w in word]
        if word not in self.vocab:
            return self.unk_id
        return self.vocab[word]
    def id_to_word(self, index):
        if isinstance(index, list):
            return [self.id_to_word(i) for i in index]
        if index < 0 or index >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        return self.reverse_vocab[index]


PRINT_STRING = """
({4:.2f} img/sec) iteration: {0:05d} loss: {1:.5f}
    caption: {2}
    actual: {3}"""


if __name__ == "__main__":


    dataset_iterator = get_dataset(FLAGS.tfrecord_file_pattern, FLAGS.image_height, FLAGS.image_width, 
        FLAGS.num_epochs, FLAGS.batch_size)
    with open(FLAGS.vocab_filename, "rb") as f:
        reverse_vocab = pkl.load(f) + ("<s>", "</s>", "<unk>")
        vocab = Vocabulary(reverse_vocab, "<s>", "</s>", "<unk>")
        vocab_table = tf.contrib.lookup.index_to_string_table_from_tensor(reverse_vocab, default_value='<unk>')
    dataset_initializer = dataset_iterator.initializer
    dataset = dataset_iterator.get_next()
        
    cnn = ResNet(global_pool=False)
    image_features = cnn(dataset["image"] / 127.5 - 1.0)
    boxes = dataset["boxes"]

    batch_size = tf.shape(image_features)[0]
    cnn_height = tf.shape(image_features)[1]
    cnn_width = tf.shape(image_features)[2]
    num_regions = tf.shape(boxes)[1]
    
    boxes = tf.expand_dims(tf.expand_dims(boxes, 3), 4)
    y_positions = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.linspace(0.0, 1.0, cnn_height), 0), 1), 3)
    x_positions = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.linspace(0.0, 1.0, cnn_width), 0), 1), 2)
    region_masks = tf.expand_dims(tf.where(
        tf.math.logical_and(tf.math.greater_equal(y_positions, boxes[:, :, 0, :, :]),  
        tf.math.logical_and(tf.math.greater_equal(x_positions, boxes[:, :, 1, :, :]), 
        tf.math.logical_and(tf.math.less_equal(y_positions, boxes[:, :, 2, :, :]), 
        tf.math.less_equal(x_positions, boxes[:, :, 3, :, :])))), 
        tf.ones([batch_size, num_regions, cnn_height, cnn_width]), 
        tf.zeros([batch_size, num_regions, cnn_height, cnn_width])), 4)

    region_features = tf.reduce_sum(
        tf.expand_dims(image_features, 1) * region_masks, [2, 3]) / (tf.reduce_sum(
            region_masks, [2, 3]) + 1e-9)

    upd = UpDownCell(1024)
    captioner = ImageCaptioner(upd, vocab, np.random.normal(0, 0.001, [len(reverse_vocab), 1024]))
    logits, ids = captioner(lengths=tf.reduce_sum(dataset["indicator"], axis=1), 
        mean_image_features=tf.reduce_mean(image_features, [1, 2]), 
        mean_object_features=region_features, 
        seq_inputs=dataset["input_seq"])
    tf.losses.sparse_softmax_cross_entropy(dataset["target_seq"], logits, weights=dataset["indicator"])
    loss = tf.losses.get_total_loss()
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(0.001, global_step, 5000, 0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    learning_step = optimizer.minimize(loss, var_list=captioner.variables, global_step=global_step)
    captioner_saver = tf.train.Saver(var_list=captioner.variables + [global_step])
    tf.gfile.MakeDirs("./up_down_ckpts/")
    with tf.Session() as sess:
        resnet_saver = tf.train.Saver(var_list=cnn.variables)
        resnet_saver.restore(sess, 'resnet_v2_101.ckpt')
        sess.run(dataset_initializer)
        sess.run(tf.tables_initializer())
        sess.run(tf.variables_initializer(optimizer.variables()))

        latest_checkpoint = tf.train.latest_checkpoint("./up_down_ckpts/")
        if latest_checkpoint is not None:
            captioner_saver.restore(sess, latest_checkpoint)
        else:
            sess.run(tf.variables_initializer(captioner.variables + [global_step]))

        captioner_saver.save(sess, "./up_down_ckpts/model.ckpt", global_step=global_step)
        last_save = time.time()
        for i in itertools.count():
            time_start = time.time()
            try:
                _, np_loss = sess.run([learning_step, loss])
                #caption = sess.run(tf.strings.reduce_join(vocab_table.lookup(tf.cast(ids, tf.int64)), axis=1, separator=" "))
            except Exception as e:
                print(e)
                break
            print("Finished iteration {0} with ({1:.2f} img/sec) loss was {2:.5f}".format(
                i, FLAGS.batch_size / (time.time() - time_start), np_loss))
            #print(caption[0])
            new_save = time.time()
            if new_save - last_save > 3600: # save the model every hour
                captioner_saver.save(sess, "./up_down_ckpts/model.ckpt", global_step=global_step)
                last_save = new_save
        captioner_saver.save(sess, "./up_down_ckpts/model.ckpt", global_step=global_step)
        print("Finishing training.")