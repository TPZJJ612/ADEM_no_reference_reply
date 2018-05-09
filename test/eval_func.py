#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from ADEM.model_adem import *
from tensorflow.contrib import learn
# Parameters
# ==================================================

# Data Parameters
# tf.flags.DEFINE_string("positive_data_file", "./data/functionbody3/test_test.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/functionbody3/test_notest.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("positive_data_file", "./data.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/functionbody3/test_notest.txt", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

tf.flags.DEFINE_string("checkpoint_dir", ".\\runs\\", "Checkpoint directory from training run")

tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    # has label
    # x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file)
    # y_test = np.argmax(y_test, axis=1)
else:
    # x_raw = ["a masterpiece four years in the making", "everything is off."]
    # x_raw.append("great movie")
    # x_raw, y_test = data_helpers.load_data_only(FLAGS.positive_data_file, FLAGS.negative_data_file)
    x_raw, y_test = data_helpers.load_data_only(FLAGS.positive_data_file)

    y_test = np.argmax(y_test, axis=1)
    # y_test = [1, 0]
    # y_test.append(0)

# Map data into vocabulary
vocab_path = "./vocab"
# vocab_path = "D:\ReadBehavioursLog\cnntextclassification01\\runs\\1502439002\\vocab"
print(vocab_path)
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
tmp_str1 = FLAGS.checkpoint_dir
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        tmp_str = "{}.meta".format(checkpoint_file)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name

        context = graph.get_operation_by_name("input_placeholder/context_place").outputs[0]
        model_response_place = graph.get_operation_by_name("input_placeholder/model_response_place").outputs[0]
        reference_response_place = graph.get_operation_by_name("input_placeholder/reference_response_place").outputs[0]

        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("score/model_score1").outputs[0]
        # scores = graph.get_operation_by_name("output/scores").outputs[0]
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        # batches = data_helpers.batch_iter(list(zip(x_train, y_train))
        #                                   , FLAGS.batch_size
        #                                   , FLAGS.num_epochs
        #                                   )
        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            # x_batch = zip(*x_test_batch)
            # x_batch = np.array(list(x_batch))
            # y_batch = np.array(list(y_batch))

            batch_predictions = sess.run(scores, {context: x_test_batch
                , model_response_place: x_test_batch
                , reference_response_place: x_test_batch}
                                         )

            # batch_scores = sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            # tmp_score = scores
            all_predictions = np.concatenate([all_predictions, batch_predictions])
print(all_predictions)
print(len(all_predictions))
exit(0)
# fliter some dirty sentence:

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw),y_test,all_predictions ))






