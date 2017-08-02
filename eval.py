#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
from knx.util.logging import Timing

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================
# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", None, "Checkpoint directory from training run")
tf.flags.DEFINE_string("testdir", None, "Test directory")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
classes = ['Adult', 'Car_accident', 'Death_tragedy', 'Hate_speech', 'Religion', 'Safe']

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
x_raw, y_test, test_filenames = data_helpers.load_data_and_labels(FLAGS.testdir, used_onehot=False, return_filenames=True)
#Convert string label into int label
y_test = [classes.index(e) for e in y_test]

# Map data into vocabulary
import cPickle as pkl
with Timing('Loading vocab & transform test x_raw...\n'):
    vocab_processor = pkl.load(open('vocab.pkl', 'rb'))
    x_test = np.array(list(vocab_processor.transform(x_raw)))

with Timing("\nEvaluating...\n"):
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None

            for x_test_batch in batches:
                batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
                probabilities = softmax(batch_predictions_scores[1])
                if all_probabilities is not None:
                    all_probabilities = np.concatenate([all_probabilities, probabilities])
                else:
                    all_probabilities = probabilities

print len(all_predictions), len(y_test)
print y_test[:5]
print all_predictions[:5]
preds = all_predictions
# Print accuracy if y_test is defined
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

with Timing("Evaluating model ..."):
    f1 = f1_score(y_test, preds, pos_label=None, average='macro')
    precision = precision_score(y_test, preds, pos_label=None, average='macro')
    recall = recall_score(y_test, preds, pos_label=None, average='macro')

print "F1: " + str(f1)
print "Precision: " + str(precision)
print "Recall: " + str(recall)
print(metrics.classification_report(y_test, preds))
print(metrics.confusion_matrix(y_test, preds))

f = open('error-log.txt', 'wb')
for i in range(len(preds)):
    print y_test[i], preds[i]
    true_label = int(y_test[i])
    predict_label = int(preds[i])
    f.write('True Label, Predict label, filename\n') 
    if true_label != predict_label:
        f.write('{0} {1} {2}\n'.format(classes[true_label], classes[predict_label], test_filenames[i]))
f.close()