#!/usr/bin/env python
# coding=utf-8
# 解释头务必要加，否则会报错
import numpy as np
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from ADEM.model_adem import *

from tensorflow.contrib import learn
import jieba
# import xlrd

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                            'Steps to validate and print loss')
tf.app.flags.DEFINE_string("log_dir", "/tmp/check_point", "log dir for tensorboard")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


##################################
###
##
#


# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("label_data_file", "./data.txt", "Data source for the label data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5,6,7,8,9", "Comma-separated filter sizes (default: '3,4,5')")

tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

print("preprocess data...")
# before loading data cut words........txt file in style: sample1 \t label

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.label_data_file)
# Build vocabulary

max_document_length = max([len(x.split(" ")) for x in x_text])
# print(x_text)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x = np.array(list(vocab_processor.fit_transform(x_text)))
y = np.array(y)
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))

x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set train:9596   test:1066
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# print(dev_sample_index)
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# learning_rate = 0.1
max_grad_norm = 5
context_dim = 94
model_response_dim = 94
reference_response_dim = 94
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        global_step = tf.Variable(0, name="global_step", trainable=False)

        model = ADEM( context_dim
                     , model_response_dim
                     , reference_response_dim
                     , learning_rate
                     , max_grad_norm
                     )

        # save models:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())

        # write vocabulary:
        vocab_processor.save("vocab")
        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train))
                                          , FLAGS.batch_size
                                          , FLAGS.num_epochs
                                          )
        # Training loop. For each batch...
        batch_count = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(list(x_batch))
            y_batch = np.array(list(y_batch))
            prediction, loss = model.train_on_single_batch(
                  sess
                , context = x_batch
                , model_response = x_batch
                , reference_response = x_batch
                , human_score = y_batch
            )
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                print("loss_value : %s", loss)
            if current_step % FLAGS.checkpoint_every == 0:
                print("Saved model checkpoint")
                path = saver.save(sess, "./runs/", global_step = current_step)

